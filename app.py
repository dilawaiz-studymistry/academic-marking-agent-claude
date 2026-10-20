import streamlit as st
import fitz  # PyMuPDF engine
import json
import re
import base64
import anthropic

# 1. Page Configuration
st.set_page_config(page_title="Academic Marking Agent (Claude)", layout="wide")
st.title("🎓 AI Academic Marking Agent (Powered by Claude)")
st.markdown("Upload a **Student Answer Script** and a **Marking Scheme** to get visual score annotations, an Excel summary matrix, and a stamped PDF.")

# 2. Retrieve Anthropic API Key safely from Secrets
if "ANTHROPIC_API_KEY" in st.secrets:
    api_key = st.secrets["ANTHROPIC_API_KEY"]
else:
    st.error("Missing ANTHROPIC_API_KEY in Streamlit Secrets. Please add it to your Streamlit Cloud configuration.")
    st.stop()

# 3. System Instructions with Strict Absence Rules & Temperature Zero Rules
SYSTEM_INSTRUCTIONS = """
You are a precise Academic Data Formatting Agent. 

You will analyze images of a handwritten or typed Student Answer Script alongside a master Marking Scheme. Your goal is to map out exact scores per question and output structured data meant for spreadsheet population.

CRITICAL OPERATIONAL & ACCURACY RULES:
- DETERMINISTIC ACCURACY: Evaluate strictly what is physically written on the page. Do NOT hallucinate or guess missing answers.
- ABSENCE RULE: If a student has skipped a question or sub-question, explicitly mark it as 'Not Attempted' with 0 marks awarded. Do NOT infer answers from surrounding text.
- Track student responses solely by their structural identifier (e.g., Q1, Q2, Q4(b)i, Q4(b)ii).

OUTPUT REQUIREMENTS:

SECTION 1: VISUAL ANNOTATION LIST
Produce a short list detailing what marks to physically draw over each question on the student's paper. Use this exact syntax:
- [Question Number]: Write "[Score Awarded] / [Max Score]"

SECTION 2: EXCEL SPREADSHEET MATRIX
Provide a clean Markdown table containing your granular evaluations. Format your table with these exact columns:
| Question # | Max Marks | Marks Awarded | Deductions | Deduction Reason / Feedback Summary |

SECTION 3: JSON STRUCTURE FOR PDF STAMPING
AT THE VERY END OF YOUR RESPONSE, provide a raw JSON array. For each evaluated item (including all sub-parts), provide:
- 'page': page index (1-based)
- 'target_text': exact question heading label as written (e.g. "4(b)i", "4(b)ii", "ii)")
- 'y': estimated vertical position (0 to 800 scale, where 0 is top margin and 800 is bottom margin) corresponding EXACTLY to where that specific sub-question answer is written.
- 'score_text': score summary text to stamp (e.g. "4(b)ii: 2/2")

Example JSON structure required at the very end:
[
  {
    "page": 1,
    "target_text": "4(b)i",
    "y": 140,
    "score_text": "4(b)i: 1/1"
  },
  {
    "page": 1,
    "target_text": "4(b)ii",
    "y": 280,
    "score_text": "4(b)ii: 2/2"
  }
]
"""

# Helper function to convert PDF pages to Base64 Image payloads for Claude
def pdf_bytes_to_base64_images(pdf_bytes):
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    images_base64 = []
    for page in doc:
        pix = page.get_pixmap(dpi=150)
        img_bytes = pix.tobytes("jpeg")
        encoded = base64.b64encode(img_bytes).decode("utf-8")
        images_base64.append(encoded)
    return images_base64

# 4. Helper Function: Hybrid Anchor PDF Annotation
def burn_annotations_to_pdf(pdf_bytes, json_data):
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    
    try:
        annotations = json.loads(json_data)
        used_positions = {}
        
        for item in annotations:
            page_num = item.get("page", 1) - 1
            if 0 <= page_num < len(doc):
                page = doc[page_num]
                target_text = str(item.get("target_text", "")).strip()
                score_text = str(item.get("score_text", "")).strip()
                raw_y = float(item.get("y", 100))
                
                page_width = page.rect.width
                page_height = page.rect.height
                x = max(page_width - 170.0, 50.0)
                found_y = None
                
                if target_text:
                    matches = page.search_for(target_text)
                    if not matches and "(" in target_text:
                        sub_part = target_text[target_text.find("("):]
                        matches = page.search_for(sub_part)
                    if matches:
                        found_y = matches[0].y1 + 2.0

                if found_y is None:
                    found_y = raw_y
                
                if page_num not in used_positions:
                    used_positions[page_num] = []
                
                for existing_y in used_positions[page_num]:
                    if abs(existing_y - found_y) < 10.0:
                        found_y += 14.0
                        
                used_positions[page_num].append(found_y)
                y = min(max(found_y, 25.0), page_height - 25.0)
                
                try:
                    page.insert_text(
                        fitz.Point(x, y),
                        score_text,
                        fontsize=11,
                        color=(0.85, 0, 0),
                        fontname="helv"
                    )
                except Exception:
                    rect = fitz.Rect(x, y, x + 150, y + 20)
                    annot = page.add_freetext_annot(rect, score_text, fontsize=11, text_color=(0.85, 0, 0))
                    annot.update()
                    
    except Exception as e:
        st.warning(f"Could not parse automatic PDF overlay: {e}")

    return doc.tobytes()

# 5. User Interface
col1, col2 = st.columns(2)
with col1:
    student_file = st.file_uploader("Upload Student Script (PDF)", type=["pdf"])
with col2:
    scheme_file = st.file_uploader("Upload Marking Scheme (PDF)", type=["pdf"])

# 6. Execution Logic via Anthropic Claude API
if st.button("🚀 Run Assessment Marking (Claude)"):
    if not student_file or not scheme_file:
        st.warning("Please upload both the Student Answer Script and the Marking Scheme.")
    else:
        with st.spinner("Analyzing document pages using Claude 3.5 Sonnet..."):
            try:
                client = anthropic.Anthropic(api_key=api_key)
                
                student_bytes = student_file.read()
                scheme_bytes = scheme_file.read()
                
                student_imgs = pdf_bytes_to_base64_images(student_bytes)
                scheme_imgs = pdf_bytes_to_base64_images(scheme_bytes)
                
                # Assemble multimodal content block for Anthropic API
                content_payload = []
                
                content_payload.append({"type": "text", "text": "=== STUDENT ANSWER SCRIPT PAGES ==="})
                for img_b64 in student_imgs:
                    content_payload.append({
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/jpeg",
                            "data": img_b64
                        }
                    })
                    
                content_payload.append({"type": "text", "text": "=== MARKING SCHEME PAGES ==="})
                for img_b64 in scheme_imgs:
                    content_payload.append({
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/jpeg",
                            "data": img_b64
                        }
                    })
                    
                content_payload.append({
                    "type": "text",
                    "text": "Grade the student script against the marking scheme according to system instructions."
                })

                # Call Claude API with temperature=0.0 for strict consistency
                message = client.messages.create(
                    model="claude-3-5-sonnet-20241022",
                    max_tokens=4000,
                    temperature=0.0,
                    system=SYSTEM_INSTRUCTIONS,
                    messages=[
                        {"role": "user", "content": content_payload}
                    ]
                )
                
                response_text = message.content[0].text

                if response_text:
                    st.success("Marking Complete!")
                    st.markdown("---")
                    
                    st.markdown(response_text)
                    
                    json_match = re.search(r"\[\s*\{.*\}\s*\]", response_text, re.DOTALL)
                    if json_match:
                        json_str = json_match.group(0)
                        annotated_pdf_bytes = burn_annotations_to_pdf(student_bytes, json_str)
                        
                        st.download_button(
                            label="📥 Download Marked Student PDF",
                            data=annotated_pdf_bytes,
                            file_name=f"Marked_Claude_{student_file.name}",
                            mime="application/pdf"
                        )
                
            except Exception as e:
                st.error(f"Error processing assessment with Claude: {str(e)}")
