import base64
import io
import os
import pathlib
import sys
import urllib.request
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st
from fastai.vision.all import PILImage, load_learner
from openai import OpenAI
from PIL import Image, ImageFilter


if sys.platform == "win32":
    pathlib.PosixPath = pathlib.WindowsPath


APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent
MODEL_PATH = PROJECT_ROOT / "convnextv2_thev1_best_for_good.pkl"
IMAGE_DIR = PROJECT_ROOT / "Image"
CLASS_ORDER = ["Blood", "Diarrhea", "Green", "Mucus", "Normal", "Yellow"]

UI = {
    "th": {
        "language_name": "ไทย",
        "hero_title": "Stool Image Review",
        "hero_text": "เว็บต้นแบบสำหรับทดลองจำแนกลักษณะอุจจาระจากภาพ พร้อมข้อมูลประกอบแบบอ่านง่ายก่อนตัดสินใจดูรายละเอียดภาพจริง",
        "status": "เริ่มด้วยรูปตัวอย่าง + เบลอภาพอัตโนมัติ",
        "notice": "ข้อมูลทั้งหมดเป็นคำแนะนำเบื้องต้นเพื่อการศึกษา ไม่ใช่การวินิจฉัยทางการแพทย์ หากมีอาการรุนแรง เป็นต่อเนื่อง หรือกังวลใจ ควรปรึกษาแพทย์หรือผู้เชี่ยวชาญ",
        "control": "การตั้งค่า",
        "language": "ภาษา",
        "source": "แหล่งรูปภาพ",
        "sample_mode": "ใช้รูปตัวอย่าง",
        "upload_mode": "อัปโหลดรูป",
        "blur": "Blur รูปภาพ",
        "sidebar_note": "เปิดแอปครั้งแรกจะใช้รูปตัวอย่างและเบลอภาพไว้ก่อน กดปิด Blur เพื่อดูภาพจริง",
        "choose_image": "เลือกรูปภาพ",
        "choose_class": "เลือกคลาสตัวอย่าง",
        "choose_sample": "เลือกรูป",
        "upload_file": "อัปโหลดไฟล์รูปภาพ",
        "uploaded_caption": "รูปที่อัปโหลด",
        "no_samples": "ไม่พบโฟลเดอร์รูปตัวอย่าง",
        "select_to_start": "เลือกรูปภาพเพื่อเริ่มวิเคราะห์",
        "result_title": "ผลลัพธ์และรายละเอียด",
        "no_image": "ยังไม่มีรูปภาพสำหรับวิเคราะห์",
        "analyze": "วิเคราะห์ภาพ",
        "analyzing": "กำลังวิเคราะห์ภาพ...",
        "press_analyze": "กดปุ่มวิเคราะห์ภาพเพื่อดูผลจากโมเดลและข้อมูลประกอบ",
        "model_result": "ผลการวิเคราะห์จากโมเดล",
        "confidence": "ความมั่นใจ",
        "rank": "อันดับ",
        "probability": "ความน่าจะเป็น (%)",
        "load_model": "กำลังโหลดโมเดล...",
        "model_error": "โหลดโมเดลไม่สำเร็จ",
        "missing_model": "ไม่พบไฟล์โมเดล",
        "causes": "1. สาเหตุที่เป็นไปได้",
        "risks": "2. ความเสี่ยงหรือโรคที่อาจเกี่ยวข้อง",
        "care": "3. คำแนะนำเบื้องต้นและการดูแลตัวเอง",
        "support_title": "ถามต่อเกี่ยวกับอาการและผลวิเคราะห์",
        "support_intro": "พิมพ์อาการที่พบ ระยะเวลาที่เป็น หรือคำถามเกี่ยวกับผลวิเคราะห์ เพื่อให้ OpenAI ช่วยอธิบายเพิ่มเติม",
        "support_quick": "ตัวอย่างคำถาม",
        "support_input": "พิมพ์อาการหรือคำถามของคุณ",
        "open_chat": "ถามต่อกับ OpenAI",
        "support_reset": "ล้างบทสนทนา",
        "support_welcome": "สวัสดีครับ ผมช่วยตอบคำถามเกี่ยวกับการใช้งานแอป การอ่านผล และข้อควรระวังเบื้องต้นได้",
        "support_disclaimer": "แชทนี้ใช้ OpenAI เพื่อช่วยอธิบายข้อมูลทั่วไปและผลวิเคราะห์เบื้องต้น ไม่ใช่การวินิจฉัยโรคหรือคำแนะนำทางการแพทย์เฉพาะบุคคล",
        "support_api_missing": "ยังไม่ได้ตั้งค่า OPENAI_API_KEY คุณยังพิมพ์เก็บคำถามไว้ได้ แต่ระบบจะยังไม่ตอบกลับ",
        "support_api_error": "เรียก OpenAI ไม่สำเร็จ กรุณาตรวจสอบ API key หรือการเชื่อมต่อ",
        "quick_blur": "ผลนี้แปลว่าอะไร",
        "quick_result": "มีอาการแบบนี้ควรกังวลไหม",
        "quick_medical": "ควรพบแพทย์เมื่อไร",
    },
    "en": {
        "language_name": "English",
        "hero_title": "Stool Image Review",
        "hero_text": "A prototype web app for classifying stool images, with readable guidance before users decide to reveal the original image.",
        "status": "Sample-first + Blur default",
        "notice": "All information is educational guidance only and is not a medical diagnosis. If symptoms are severe, persistent, or worrying, consult a doctor or qualified health professional.",
        "control": "Settings",
        "language": "Language",
        "source": "Image source",
        "sample_mode": "Sample images",
        "upload_mode": "Upload image",
        "blur": "Blur image",
        "sidebar_note": "The app starts with sample images and blurs the preview by default. Turn Blur off to reveal the original image.",
        "choose_image": "Choose image",
        "choose_class": "Choose sample class",
        "choose_sample": "Choose sample",
        "upload_file": "Upload image file",
        "uploaded_caption": "Uploaded image",
        "no_samples": "Sample image folder was not found",
        "select_to_start": "Choose an image to start analysis",
        "result_title": "Result and Details",
        "no_image": "No image is available for analysis",
        "analyze": "Analyze image",
        "analyzing": "Analyzing image...",
        "press_analyze": "Press Analyze image to view model output and supporting information",
        "model_result": "Model Analysis Result",
        "confidence": "Confidence",
        "rank": "Rank",
        "probability": "Probability (%)",
        "load_model": "Loading model...",
        "model_error": "Failed to load model",
        "missing_model": "Model file not found",
        "causes": "1. Possible Causes",
        "risks": "2. Possible Risks or Related Conditions",
        "care": "3. Basic Advice and Self-care",
        "support_title": "Ask about symptoms and this result",
        "support_intro": "Describe symptoms, duration, or questions about the analysis so OpenAI can explain the result in more detail.",
        "support_quick": "Example questions",
        "support_input": "Type your symptoms or question",
        "open_chat": "Ask OpenAI",
        "support_reset": "Clear conversation",
        "support_welcome": "Hi. I can help with app usage, result interpretation, and basic safety notes.",
        "support_disclaimer": "This chat uses OpenAI to explain general information and the analysis result. It is not a diagnosis or personalized medical advice.",
        "support_api_missing": "OPENAI_API_KEY is not configured. You can still type messages, but the assistant will not reply yet.",
        "support_api_error": "The OpenAI request failed. Please check the API key or connection.",
        "quick_blur": "What does this result mean?",
        "quick_result": "Should I worry about these symptoms?",
        "quick_medical": "When should I see a doctor?",
    },
}

CLASS_DETAILS = {
    "Blood": {
        "tone": "#9f4f4f",
        "soft": "#fbf5f4",
        "th": {
            "label": "มีเลือดปน",
            "summary": "ภาพถูกจัดอยู่ในกลุ่มที่อาจมีสีแดงหรือเลือดปน ควรดูร่วมกับอาการ ปริมาณเลือด ความถี่ และประวัติการกินอาหารสีแดง",
            "causes": [
                "การระคายเคืองบริเวณทวารหนัก เช่น แผลปริ ริดสีดวง หรือการเบ่งถ่ายแรง",
                "การอักเสบหรือติดเชื้อในลำไส้ ซึ่งอาจทำให้ถ่ายเหลว ปวดท้อง หรือมีมูกเลือดร่วมด้วย",
                "อาหารหรือสีผสมอาหารบางชนิดอาจทำให้สีอุจจาระดูแดงคล้ายเลือดได้",
                "เลือดออกจากทางเดินอาหารส่วนล่าง เช่น ลำไส้ใหญ่หรือไส้ตรง โดยเฉพาะถ้าเห็นเลือดสดชัดเจน",
            ],
            "risks": [
                "ริดสีดวงทวาร แผลปริขอบทวาร หรือการระคายเคืองจากท้องผูก",
                "ลำไส้อักเสบ การติดเชื้อในทางเดินอาหาร หรือภาวะลำไส้แปรปรวนบางรูปแบบ",
                "ถ้ามีเลือดจำนวนมาก หน้ามืด ไข้สูง ปวดท้องรุนแรง หรือน้ำหนักลด ควรพบแพทย์เร็ว",
            ],
            "care": [
                "สังเกตสี ปริมาณเลือด ความถี่ และถ่ายรูปเก็บไว้หากต้องไปพบแพทย์",
                "ดื่มน้ำให้เพียงพอ เพิ่มไฟเบอร์ และหลีกเลี่ยงการเบ่งถ่ายแรง",
                "ถ้าพบเลือดซ้ำหลายครั้ง มีอาการปวดมาก หรือมีอาการผิดปกติร่วม ควรปรึกษาแพทย์",
            ],
        },
        "en": {
            "label": "Blood present",
            "summary": "The image is classified as possibly containing red coloration or blood. Interpret this together with symptoms, amount, frequency, and recent red-colored foods.",
            "causes": [
                "Irritation around the anus, such as fissures, hemorrhoids, or straining during bowel movements.",
                "Inflammation or infection in the intestines, which may also cause loose stool, abdominal pain, or mucus with blood.",
                "Certain foods or food dyes can make stool appear red and look similar to blood.",
                "Bleeding from the lower digestive tract, such as the colon or rectum, especially if bright red blood is clearly visible.",
            ],
            "risks": [
                "Hemorrhoids, anal fissures, or irritation related to constipation.",
                "Intestinal inflammation, gastrointestinal infection, or some patterns of irritable bowel symptoms.",
                "Seek medical care promptly if bleeding is heavy, dizziness occurs, fever is high, pain is severe, or weight loss is present.",
            ],
            "care": [
                "Observe color, amount, frequency, and keep a record or photo if medical consultation is needed.",
                "Drink enough water, increase fiber gradually, and avoid excessive straining.",
                "Consult a doctor if blood appears repeatedly, pain is significant, or other concerning symptoms occur.",
            ],
        },
    },
    "Diarrhea": {
        "tone": "#9a6a3d",
        "soft": "#fbf7ef",
        "th": {
            "label": "ท้องเสีย",
            "summary": "ภาพถูกจัดอยู่ในกลุ่มอุจจาระเหลวหรือถ่ายเหลว ซึ่งอาจสัมพันธ์กับอาหาร การติดเชื้อ หรือการระคายเคืองของลำไส้",
            "causes": [
                "อาหารไม่สะอาด อาหารรสจัด นม คาเฟอีน แอลกอฮอล์ หรืออาหารที่ร่างกายย่อยยาก",
                "การติดเชื้อไวรัส แบคทีเรีย หรือพยาธิในทางเดินอาหาร",
                "ผลข้างเคียงจากยา เช่น ยาปฏิชีวนะ ยาระบาย หรืออาหารเสริมบางชนิด",
                "ความเครียดหรือภาวะลำไส้ไวต่อสิ่งกระตุ้น ทำให้ลำไส้เคลื่อนไหวเร็วขึ้น",
            ],
            "risks": [
                "ภาวะขาดน้ำและเกลือแร่ โดยเฉพาะถ้าถ่ายบ่อย อาเจียน หรือกินน้ำได้น้อย",
                "อาหารเป็นพิษหรือลำไส้อักเสบ หากมีไข้ ปวดบิด หรือมีมูกเลือดร่วม",
                "ถ้าท้องเสียต่อเนื่องหลายวัน อ่อนเพลียมาก หรือปัสสาวะน้อย ควรพบแพทย์",
            ],
            "care": [
                "จิบน้ำหรือเกลือแร่ ORS บ่อย ๆ และหลีกเลี่ยงน้ำหวานจัด",
                "กินอาหารอ่อน ย่อยง่าย เช่น ข้าวต้ม กล้วย ขนมปัง หรือซุปใส",
                "หลีกเลี่ยงนม ของทอด แอลกอฮอล์ และอาหารรสจัดจนกว่าอาการจะดีขึ้น",
            ],
        },
        "en": {
            "label": "Diarrhea",
            "summary": "The image is classified as loose or watery stool, which may be related to food, infection, or intestinal irritation.",
            "causes": [
                "Contaminated food, spicy food, dairy, caffeine, alcohol, or foods that are difficult to digest.",
                "Viral, bacterial, or parasitic infection in the digestive tract.",
                "Medication side effects, such as antibiotics, laxatives, or some supplements.",
                "Stress or bowel sensitivity that increases intestinal movement.",
            ],
            "risks": [
                "Dehydration and electrolyte loss, especially with frequent stool, vomiting, or poor fluid intake.",
                "Food poisoning or intestinal inflammation if fever, cramping pain, or mucus/blood is present.",
                "See a doctor if diarrhea lasts several days, weakness is severe, or urination becomes very low.",
            ],
            "care": [
                "Sip water or oral rehydration solution often, and avoid very sweet drinks.",
                "Choose soft, easy-to-digest foods such as rice porridge, bananas, bread, or clear soup.",
                "Avoid dairy, fried foods, alcohol, and spicy food until symptoms improve.",
            ],
        },
    },
    "Green": {
        "tone": "#527a61",
        "soft": "#f3f8f4",
        "th": {
            "label": "สีเขียว",
            "summary": "ภาพถูกจัดอยู่ในกลุ่มสีเขียว อาจเกิดจากอาหาร สีผสมอาหาร หรือการเคลื่อนตัวของลำไส้ที่เร็วขึ้นจนสีน้ำดีเปลี่ยนไม่ทัน",
            "causes": [
                "กินผักใบเขียว สาหร่าย เครื่องดื่มสีเขียว หรืออาหารที่มีสีผสมอาหาร",
                "ลำไส้เคลื่อนตัวเร็ว เช่น หลังท้องเสีย ทำให้น้ำดีที่มีสีเขียวยังเปลี่ยนเป็นสีน้ำตาลไม่สมบูรณ์",
                "อาหารเสริมธาตุเหล็กหรือยาบางชนิดอาจเปลี่ยนสีอุจจาระได้",
                "ในบางกรณีอาจเกี่ยวข้องกับการติดเชื้อหรือการระคายเคืองของทางเดินอาหาร",
            ],
            "risks": [
                "ส่วนใหญ่ไม่อันตรายถ้าเกิดหลังรับประทานอาหารสีเขียวและไม่มีอาการอื่น",
                "ควรระวังถ้ามีท้องเสีย ไข้ ปวดท้อง อาเจียน หรือถ่ายผิดปกติต่อเนื่อง",
                "หากสีเขียวเกิดซ้ำโดยไม่สัมพันธ์กับอาหารหรือยา ควรติดตามอาการเพิ่ม",
            ],
            "care": [
                "ทบทวนอาหาร ยา และอาหารเสริมที่กินใน 24-48 ชั่วโมงที่ผ่านมา",
                "ดื่มน้ำให้เพียงพอ โดยเฉพาะถ้ามีถ่ายเหลวร่วม",
                "หากมีอาการผิดปกติร่วม หรือเป็นต่อเนื่องหลายวัน ควรปรึกษาผู้เชี่ยวชาญ",
            ],
        },
        "en": {
            "label": "Green stool",
            "summary": "The image is classified as green stool. This can be caused by food, food coloring, or faster bowel movement before bile fully changes color.",
            "causes": [
                "Green leafy vegetables, seaweed, green drinks, or foods containing green coloring.",
                "Rapid intestinal movement, such as after diarrhea, leaving bile with a green color.",
                "Iron supplements or some medicines may change stool color.",
                "In some cases, infection or irritation in the digestive tract may be involved.",
            ],
            "risks": [
                "Often low risk if it follows green food intake and there are no other symptoms.",
                "Pay attention if diarrhea, fever, abdominal pain, vomiting, or persistent abnormal stool occurs.",
                "If green stool repeats without a food or medication link, continue monitoring symptoms.",
            ],
            "care": [
                "Review food, medicine, and supplements taken in the past 24-48 hours.",
                "Drink enough water, especially if loose stool is also present.",
                "Consult a health professional if symptoms persist for several days or other symptoms appear.",
            ],
        },
    },
    "Mucus": {
        "tone": "#4f7d86",
        "soft": "#f1f7f8",
        "th": {
            "label": "มีมูกปน",
            "summary": "ภาพถูกจัดอยู่ในกลุ่มที่อาจมีมูกปน มูกเล็กน้อยพบได้บ้าง แต่ถ้ามีมากหรือเกิดซ้ำควรสังเกตอาการร่วม",
            "causes": [
                "การระคายเคืองของลำไส้จากอาหาร ความเครียด หรือการขับถ่ายที่ผิดปกติ",
                "ท้องผูกหรือท้องเสียทำให้เยื่อบุลำไส้สร้างมูกมากขึ้น",
                "การติดเชื้อในลำไส้ อาจมีมูกพร้อมถ่ายเหลว ปวดท้อง หรือมีไข้",
                "ภาวะลำไส้อักเสบบางชนิดอาจทำให้มีมูก เลือด หรือปวดท้องเรื้อรัง",
            ],
            "risks": [
                "ถ้ามูกมีเล็กน้อยและหายเอง อาจเป็นการระคายเคืองชั่วคราว",
                "ควรระวังถ้ามีมูกมาก มีกลิ่นผิดปกติ มีเลือด ไข้ หรือปวดท้องต่อเนื่อง",
                "หากเป็นซ้ำบ่อย น้ำหนักลด หรือถ่ายผิดปกติเรื้อรัง ควรพบแพทย์",
            ],
            "care": [
                "จดบันทึกความถี่ ลักษณะมูก อาหาร และอาการร่วม",
                "ดื่มน้ำ พักผ่อน และกินอาหารย่อยง่ายช่วงที่ลำไส้ระคายเคือง",
                "หลีกเลี่ยงการใช้ยาฆ่าเชื้อหรือยาหยุดถ่ายเองโดยไม่ปรึกษาผู้เชี่ยวชาญ",
            ],
        },
        "en": {
            "label": "Mucus present",
            "summary": "The image is classified as possibly containing mucus. Small amounts can occur, but larger or repeated mucus should be interpreted with symptoms.",
            "causes": [
                "Intestinal irritation from food, stress, or abnormal bowel patterns.",
                "Constipation or diarrhea can increase mucus production from the intestinal lining.",
                "Intestinal infection may cause mucus along with loose stool, abdominal pain, or fever.",
                "Some inflammatory bowel conditions may cause mucus, blood, or ongoing abdominal pain.",
            ],
            "risks": [
                "Small amounts that resolve quickly may be temporary irritation.",
                "Be cautious if mucus is frequent, foul-smelling, bloody, associated with fever, or linked to persistent pain.",
                "See a doctor if it happens often, weight loss occurs, or bowel changes become chronic.",
            ],
            "care": [
                "Record frequency, mucus appearance, food intake, and associated symptoms.",
                "Hydrate, rest, and choose easy-to-digest foods while the bowel is irritated.",
                "Avoid self-prescribing antibiotics or anti-diarrhea medicine without professional advice.",
            ],
        },
    },
    "Normal": {
        "tone": "#496f8f",
        "soft": "#f2f6f9",
        "th": {
            "label": "ปกติ",
            "summary": "ภาพถูกจัดอยู่ในกลุ่มปกติ โดยทั่วไปสื่อถึงลักษณะที่ใกล้เคียงการขับถ่ายสุขภาพดี แต่ยังควรดูร่วมกับอาการของร่างกาย",
            "causes": [
                "การกินอาหารสมดุล มีไฟเบอร์ น้ำ และไขมันในระดับที่เหมาะสม",
                "การขับถ่ายสม่ำเสมอและลำไส้เคลื่อนไหวในจังหวะปกติ",
                "สีและรูปทรงอาจเปลี่ยนเล็กน้อยได้ตามอาหาร น้ำ และกิจวัตรประจำวัน",
            ],
            "risks": [
                "โดยรวมความเสี่ยงต่ำ หากไม่มีปวดท้อง เลือด มูก ไข้ หรือถ่ายผิดปกติ",
                "ยังควรสังเกตการเปลี่ยนแปลงที่เกิดขึ้นต่อเนื่อง เช่น ท้องผูกหรือถ่ายเหลวบ่อย",
                "ถ้ามีอาการผิดปกติแม้ภาพดูปกติ ควรให้ความสำคัญกับอาการจริงของผู้ใช้",
            ],
            "care": [
                "ดื่มน้ำให้เพียงพอและกินผัก ผลไม้ ธัญพืช หรือไฟเบอร์อย่างเหมาะสม",
                "ออกกำลังกายและพักผ่อนให้พอ เพื่อช่วยการเคลื่อนไหวของลำไส้",
                "ติดตามรูปแบบการขับถ่ายของตัวเอง เพราะความปกติของแต่ละคนอาจต่างกัน",
            ],
        },
        "en": {
            "label": "Normal",
            "summary": "The image is classified as normal. This generally suggests a healthier-looking stool pattern, but it should still be considered with real symptoms.",
            "causes": [
                "Balanced diet with suitable fiber, water, and fat intake.",
                "Regular bowel habits and normal intestinal movement.",
                "Color and shape can vary slightly depending on food, hydration, and daily routine.",
            ],
            "risks": [
                "Overall risk is lower if there is no pain, blood, mucus, fever, or abnormal bowel pattern.",
                "Continue watching for persistent changes such as constipation or frequent loose stool.",
                "If symptoms feel abnormal even when the image looks normal, symptoms should be prioritized.",
            ],
            "care": [
                "Drink enough water and include vegetables, fruit, grains, or fiber in a balanced way.",
                "Exercise and rest adequately to support bowel movement.",
                "Track your own bowel pattern because normal can vary from person to person.",
            ],
        },
    },
    "Yellow": {
        "tone": "#8f7a45",
        "soft": "#faf7ec",
        "th": {
            "label": "สีเหลือง",
            "summary": "ภาพถูกจัดอยู่ในกลุ่มสีเหลือง อาจเกี่ยวกับอาหาร ไขมัน การย่อยอาหาร หรือการเคลื่อนตัวของลำไส้ที่เร็วขึ้น",
            "causes": [
                "อาหารที่มีสีเหลือง สีผสมอาหาร หรืออาหารมันมากบางชนิด",
                "การย่อยและดูดซึมไขมันผิดปกติ อาจทำให้อุจจาระสีอ่อน เหลือง มัน หรือมีกลิ่นแรง",
                "ท้องเสียหรือการเคลื่อนตัวของลำไส้เร็ว ทำให้สีอุจจาระเปลี่ยน",
                "ความผิดปกติของน้ำดี ตับ ตับอ่อน หรือถุงน้ำดีเป็นสาเหตุที่ควรพิจารณาหากมีอาการร่วม",
            ],
            "risks": [
                "อาจไม่อันตรายถ้าเกิดชั่วคราวหลังอาหารและไม่มีอาการอื่น",
                "ควรระวังถ้ามีอุจจาระมันลอยน้ำ กลิ่นแรง น้ำหนักลด ตัวเหลือง ตาเหลือง หรือปวดท้องด้านขวาบน",
                "ถ้าเกิดซ้ำต่อเนื่องหรือมีอาการทางระบบย่อยอาหารชัดเจน ควรปรึกษาแพทย์",
            ],
            "care": [
                "สังเกตอาหารมัน สีอาหาร และอาการร่วมในช่วง 1-2 วันที่ผ่านมา",
                "ดื่มน้ำ กินอาหารย่อยง่าย และลดอาหารมันจัดชั่วคราว",
                "ถ้ามีตัวเหลือง ตาเหลือง ปวดท้องรุนแรง หรือถ่ายผิดปกติต่อเนื่อง ควรพบแพทย์",
            ],
        },
        "en": {
            "label": "Yellow stool",
            "summary": "The image is classified as yellow stool. This may relate to food, fat digestion, digestive changes, or faster intestinal movement.",
            "causes": [
                "Yellow-colored foods, food coloring, or some high-fat meals.",
                "Changes in fat digestion or absorption can make stool pale, yellow, greasy, or strong-smelling.",
                "Diarrhea or rapid intestinal movement can change stool color.",
                "Bile, liver, pancreas, or gallbladder issues should be considered if other symptoms are present.",
            ],
            "risks": [
                "May be low risk if temporary after food and no other symptoms are present.",
                "Pay attention if stool is greasy or floating, smell is strong, weight loss occurs, jaundice appears, or upper-right abdominal pain occurs.",
                "If it repeats or digestive symptoms are clear, consult a doctor.",
            ],
            "care": [
                "Review fatty foods, food coloring, and symptoms from the past 1-2 days.",
                "Hydrate, eat easy-to-digest foods, and temporarily reduce very fatty meals.",
                "Seek medical care if jaundice, severe abdominal pain, or persistent abnormal stool occurs.",
            ],
        },
    },
}

COLOR_MAP = {name: details["tone"] for name, details in CLASS_DETAILS.items()}


st.set_page_config(
    page_title="Stool Image Review",
    page_icon="AI",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
        :root {
            --ink: #1f2a37;
            --muted: #667784;
            --line: #d7e0e5;
            --soft: #f5f8f8;
            --surface: #ffffff;
            --accent: #5c8f87;
            --accent-strong: #3f756f;
            --accent-soft: #edf5f3;
        }
        .stApp {
            background: #f5f8f8;
        }
        .block-container {
            max-width: 1240px;
            padding-top: 1.25rem;
            padding-bottom: 2.5rem;
        }
        [data-testid="stSidebar"] {
            background: #f9fbfb;
            border-right: 1px solid var(--line);
        }
        .hero {
            display: grid;
            grid-template-columns: minmax(0, 1fr) auto;
            gap: 1rem;
            align-items: end;
            padding: 1.4rem 0 1rem;
            border-bottom: 1px solid var(--line);
            margin-bottom: 1rem;
        }
        .hero h1 {
            color: var(--ink);
            font-size: 2.35rem;
            line-height: 1.05;
            margin: 0 0 .45rem;
            letter-spacing: 0;
        }
        .hero p {
            color: var(--muted);
            font-size: 1.02rem;
            margin: 0;
            max-width: 780px;
        }
        .status-pill {
            border: 1px solid #c9dad7;
            background: var(--accent-soft);
            color: var(--accent-strong);
            padding: .48rem .72rem;
            border-radius: 999px;
            font-weight: 700;
            font-size: .86rem;
            white-space: nowrap;
        }
        .notice {
            border: 1px solid #d8d5c3;
            border-left: 5px solid #9c9272;
            background: #fbfaf4;
            color: #4f4937;
            padding: .9rem 1rem;
            border-radius: 8px;
            margin-bottom: 1rem;
        }
        .panel {
            border: 1px solid var(--line);
            background: var(--surface);
            border-radius: 8px;
            padding: 1rem;
            box-shadow: 0 10px 22px rgba(31, 42, 55, .045);
        }
        .image-frame {
            border: 1px solid var(--line);
            border-radius: 8px;
            overflow: hidden;
            background: #1f2a37;
        }
        .image-frame img {
            display: block;
            width: 100%;
            max-height: 520px;
            object-fit: contain;
            background: #1f2a37;
        }
        .result-card {
            border: 1px solid var(--line);
            border-radius: 8px;
            background: var(--surface);
            padding: 1rem;
            margin-bottom: .9rem;
        }
        .result-label {
            color: var(--muted);
            font-size: .8rem;
            text-transform: uppercase;
            letter-spacing: .04em;
            font-weight: 800;
            margin-bottom: .3rem;
        }
        .prediction {
            font-size: 1.75rem;
            line-height: 1.18;
            font-weight: 800;
            margin-bottom: .35rem;
        }
        .confidence {
            color: var(--ink);
            font-weight: 700;
            margin-bottom: .65rem;
        }
        .summary {
            color: #3b4a54;
            margin: 0;
        }
        .detail-box {
            border: 1px solid var(--line);
            background: var(--surface);
            border-radius: 8px;
            padding: 1rem;
            margin-top: .8rem;
        }
        .detail-box h3 {
            font-size: 1rem;
            color: var(--ink);
            margin: 0 0 .45rem;
        }
        .detail-box ul {
            margin: 0;
            padding-left: 1.15rem;
            color: #3b4a54;
        }
        .detail-box li {
            margin: .32rem 0;
        }
        .metric-row {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: .65rem;
            margin: .8rem 0 1rem;
        }
        .mini-metric {
            border: 1px solid var(--line);
            border-radius: 8px;
            background: var(--soft);
            padding: .75rem;
        }
        .mini-metric span {
            display: block;
            color: var(--muted);
            font-size: .78rem;
            font-weight: 700;
        }
        .mini-metric strong {
            display: block;
            color: var(--ink);
            font-size: 1rem;
            margin-top: .2rem;
        }
        .support-layout {
            display: grid;
            grid-template-columns: minmax(0, 1fr) 280px;
            gap: 1rem;
        }
        .support-note {
            border: 1px solid var(--line);
            background: #f7faf9;
            border-radius: 6px;
            padding: .85rem;
            color: #4c5f68;
            font-size: .92rem;
        }
        .quick-stack {
            display: grid;
            gap: .45rem;
        }
        div.stButton > button[kind="primary"] {
            background: var(--accent);
            border-color: var(--accent);
            color: #ffffff;
        }
        div.stButton > button[kind="primary"]:hover {
            background: var(--accent-strong);
            border-color: var(--accent-strong);
            color: #ffffff;
        }
        div.stButton > button[kind="secondary"] {
            border-color: #c9dad7;
            color: var(--accent-strong);
            background: #ffffff;
        }
        div.st-key-open_chat_fab {
            position: fixed;
            right: 1.75rem;
            bottom: 1.75rem;
            z-index: 1000;
        }
        div.st-key-open_chat_fab button {
            position: relative;
            width: 60px;
            height: 60px;
            min-width: 60px;
            max-width: 60px;
            border-radius: 999px;
            border: 0;
            background: var(--accent);
            box-shadow: 0 14px 32px rgba(63, 117, 111, .24);
            color: transparent;
            font-size: 0;
            padding: 0;
            overflow: hidden;
            display: block;
        }
        div.st-key-open_chat_fab button,
        div.st-key-open_chat_fab button:focus,
        div.st-key-open_chat_fab button:active,
        div.st-key-open_chat_fab button:focus:not(:active) {
            background-color: var(--accent) !important;
            color: transparent !important;
            border-color: transparent !important;
        }
        div.st-key-open_chat_fab button p,
        div.st-key-open_chat_fab button span,
        div.st-key-open_chat_fab button div {
            display: none;
        }
        div.st-key-open_chat_fab button:hover {
            background-color: var(--accent-strong) !important;
            border: 0;
            transform: translateY(-1px);
            box-shadow: 0 16px 36px rgba(63, 117, 111, .30);
        }
        div.st-key-open_chat_fab button::before {
            content: "";
            position: absolute;
            left: 16px;
            top: 17px;
            width: 29px;
            height: 21px;
            background: #ffffff;
            border-radius: 7px;
        }
        div.st-key-open_chat_fab button::after {
            content: "";
            position: absolute;
            left: 35px;
            top: 33px;
            width: 11px;
            height: 11px;
            background: #ffffff;
            clip-path: polygon(0 0, 100% 0, 100% 100%);
        }
        @media (max-width: 760px) {
            .hero {
                grid-template-columns: 1fr;
            }
            .hero h1 {
                font-size: 1.8rem;
            }
            .metric-row {
                grid-template-columns: 1fr;
            }
            .support-layout {
                grid-template-columns: 1fr;
            }
            div.st-key-open_chat_fab {
                right: 1rem;
                bottom: 1rem;
            }
        }
    </style>
    """,
    unsafe_allow_html=True,
)


with st.sidebar:
    language_choice = st.radio(
        "ภาษา / Language",
        ["ไทย", "English"],
        index=0,
        horizontal=True,
    )

LANG = "th" if language_choice == "ไทย" else "en"
T = UI[LANG]


@st.cache_resource(show_spinner=False)
def load_model(model_path: Path):
    """Load model from local path or download from GitHub Releases if not found."""
    
    # Try local path first
    if model_path.exists():
        return load_learner(model_path)
    
    # If not found locally, download from GitHub Releases
    model_filename = model_path.name
    model_cache_dir = Path.home() / ".streamlit_model_cache"
    model_cache_dir.mkdir(parents=True, exist_ok=True)
    
    cached_model_path = model_cache_dir / model_filename
    
    if cached_model_path.exists():
        return load_learner(cached_model_path)
    
    # Download from GitHub Releases
    try:
        st.info("⏳ Downloading model (~110MB)... This may take a minute.")
        
        github_release_url = "https://github.com/thanithpol2545/Super-AI-Hack-OPEN/releases/download/v1.0/convnextv2_thev1_best_for_good.pkl"
        
        def download_progress_hook(block_num, block_size, total_size):
            if total_size > 0:
                downloaded = block_num * block_size
                percent = min(100, int(100 * downloaded / total_size))
                if percent % 10 == 0 and percent > 0:
                    st.write(f"Downloaded: {percent}%")
        
        urllib.request.urlretrieve(github_release_url, cached_model_path, download_progress_hook)
        st.success("✅ Model downloaded successfully!")
        
        return load_learner(cached_model_path)
    
    except Exception as e:
        error_msg = f"Failed to load model: {str(e)}\nPlease ensure the model file exists or check your internet connection."
        st.error(error_msg)
        raise FileNotFoundError(error_msg)


def class_text(class_name: str):
    return CLASS_DETAILS[class_name][LANG]


def list_sample_images():
    samples = {}
    if not IMAGE_DIR.exists():
        return samples

    for class_name in CLASS_ORDER:
        class_dir = IMAGE_DIR / class_name
        if not class_dir.exists():
            continue

        images = sorted(
            path
            for path in class_dir.iterdir()
            if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
        )
        if images:
            samples[class_name] = images
    return samples


def open_pil_image(image_source):
    if isinstance(image_source, Path):
        return Image.open(image_source).convert("RGB")

    image_source.seek(0)
    image = Image.open(image_source).convert("RGB")
    image_source.seek(0)
    return image


def image_to_data_url(image: Image.Image):
    output = io.BytesIO()
    image.save(output, format="JPEG", quality=88)
    encoded = base64.b64encode(output.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def render_preview_image(image_source, caption: str, blurred: bool):
    image = open_pil_image(image_source)
    if blurred:
        image = image.filter(ImageFilter.GaussianBlur(radius=18))

    st.markdown(
        f"""
        <div class="image-frame">
            <img src="{image_to_data_url(image)}" alt="{caption}">
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.caption(caption)


def predict_image(learn, image_source):
    image = PILImage.create(image_source)
    pred_class, pred_idx, probs = learn.predict(image)
    probabilities = pd.DataFrame(
        {
            "Class": list(learn.dls.vocab),
            "Probability": [float(prob) * 100 for prob in probs],
        }
    ).sort_values("Probability", ascending=False)
    return str(pred_class), float(probs[pred_idx]), probabilities


def render_probability_chart(probabilities: pd.DataFrame):
    fig = px.bar(
        probabilities,
        x="Probability",
        y="Class",
        orientation="h",
        text=probabilities["Probability"].map(lambda value: f"{value:.1f}%"),
        color="Class",
        color_discrete_map=COLOR_MAP,
    )
    fig.update_layout(
        height=320,
        margin=dict(l=0, r=18, t=8, b=0),
        xaxis_title=T["probability"],
        yaxis_title=None,
        showlegend=False,
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(color="#3b4a54"),
    )
    fig.update_xaxes(range=[0, 100], gridcolor="#e7eef0")
    fig.update_yaxes(autorange="reversed")
    fig.update_traces(textposition="outside", cliponaxis=False)
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


def render_detail_section(title: str, items: list[str]):
    st.markdown(
        f"""
        <div class="detail-box">
            <h3>{title}</h3>
            <ul>
                {''.join(f'<li>{item}</li>' for item in items)}
            </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_static_details(pred_class: str):
    details = class_text(pred_class)
    render_detail_section(T["causes"], details["causes"])
    render_detail_section(T["risks"], details["risks"])
    render_detail_section(T["care"], details["care"])


def render_result(pred_class: str, confidence: float, probabilities: pd.DataFrame):
    style = CLASS_DETAILS.get(pred_class, CLASS_DETAILS["Normal"])
    details = class_text(pred_class) if pred_class in CLASS_DETAILS else class_text("Normal")
    top3 = probabilities.head(3)

    st.markdown(
        f"""
        <div class="result-card" style="border-left: 4px solid {style["tone"]};">
            <div class="result-label">{T["model_result"]}</div>
            <div class="prediction" style="color:{style["tone"]};">
                {details["label"]} <span style="color:#667784;">({pred_class})</span>
            </div>
            <div class="confidence">{T["confidence"]}: {confidence:.1%}</div>
            <p class="summary">{details["summary"]}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    metric_html = ""
    for index, row in enumerate(top3.itertuples(index=False), start=1):
        metric_html += (
            f'<div class="mini-metric"><span>{T["rank"]} {index}</span>'
            f"<strong>{row.Class} {row.Probability:.1f}%</strong></div>"
        )
    st.markdown(f'<div class="metric-row">{metric_html}</div>', unsafe_allow_html=True)

    render_probability_chart(probabilities)
    render_static_details(pred_class)


def get_image_source_id(image_source):
    if image_source is None:
        return None
    if isinstance(image_source, Path):
        return str(image_source.resolve())
    return f"{getattr(image_source, 'name', 'upload')}:{getattr(image_source, 'size', 0)}"


def clear_result_if_changed(source_id):
    previous_id = st.session_state.get("active_source_id")
    if previous_id != source_id:
        st.session_state.active_source_id = source_id
        st.session_state.pop("last_result", None)


def build_support_reply(message: str):
    return build_openai_support_reply(message)


def build_fallback_support_reply(message: str):
    text = message.lower()

    if any(word in text for word in ["blur", "เบลอ", "เบลอภาพ", "ปิด blur"]):
        return (
            "ปิดหรือเปิด Blur ได้จากแถบ Settings ด้านซ้าย ตัวเลือกนี้มีไว้เพื่อซ่อนภาพจริงตอนเปิดแอปครั้งแรก"
            if LANG == "th"
            else "Use the Blur image toggle in the Settings sidebar. It keeps the original image hidden on first load."
        )

    if any(word in text for word in ["confidence", "มั่นใจ", "ความมั่นใจ", "probability"]):
        return (
            "ความมั่นใจคือคะแนนที่โมเดลให้กับคลาสที่ทำนาย ยิ่งสูงยิ่งแปลว่าโมเดลเลือกคลาสนั้นชัดขึ้น แต่ไม่ใช่การยืนยันทางการแพทย์"
            if LANG == "th"
            else "Confidence is the model score for the predicted class. A higher value means the model selected that class more strongly, but it is not a medical confirmation."
        )

    if any(word in text for word in ["doctor", "medical", "แพทย์", "หมอ", "โรงพยาบาล", "เลือด", "ปวด", "ไข้"]):
        return (
            "ควรพบแพทย์หากมีเลือดออกมาก ปวดท้องรุนแรง ไข้สูง ถ่ายผิดปกติต่อเนื่องหลายวัน อ่อนเพลียมาก น้ำหนักลด หรือตัวเหลืองตาเหลือง"
            if LANG == "th"
            else "Consider seeing a doctor if there is heavy bleeding, severe abdominal pain, high fever, persistent abnormal stool, marked weakness, weight loss, or yellow skin/eyes."
        )

    if any(word in text for word in ["upload", "อัปโหลด", "รูป", "image", "sample", "ตัวอย่าง"]):
        return (
            "เลือกหน้า วิเคราะห์ภาพ แล้วเลือกใช้รูปตัวอย่างหรืออัปโหลดรูป จากนั้นกดปุ่ม วิเคราะห์ภาพ เพื่อดูผลและรายละเอียด"
            if LANG == "th"
            else "Open Image review, choose a sample or upload an image, then press Analyze image to see the result and guidance."
        )

    if any(word in text for word in ["language", "ภาษา", "english", "ไทย"]):
        return (
            "เปลี่ยนภาษาได้จากตัวเลือก ไทย / English ที่ด้านบนของ sidebar โดยค่าเริ่มต้นเป็นภาษาไทย"
            if LANG == "th"
            else "Change language with the Thai / English selector at the top of the sidebar. Thai is the default."
        )

    return (
        "ผมช่วยตอบเรื่องวิธีใช้แอป การอ่านค่าความมั่นใจ การเปิด/ปิด Blur และข้อควรระวังเบื้องต้นได้ ลองถามให้เฉพาะเจาะจงขึ้นได้ครับ"
        if LANG == "th"
        else "I can help with app usage, confidence scores, the Blur setting, and basic safety notes. Try asking a more specific question."
    )


def get_openai_api_key():
    try:
        key = st.secrets.get("OPENAI_API_KEY")
        if key:
            return key
    except Exception:
        pass
    return os.getenv("OPENAI_API_KEY")


def build_openai_support_reply(message: str):
    api_key = get_openai_api_key()
    if not api_key:
        st.info(T["support_api_missing"])
        return None

    support_language = "Thai" if LANG == "th" else "English"
    recent_messages = st.session_state.get("support_messages", [])[-8:]
    conversation = [
        {
            "role": item["role"],
            "content": item["content"],
        }
        for item in recent_messages
        if item["role"] in {"user", "assistant"}
    ]
    conversation.append({"role": "user", "content": message})

    result_context = get_analysis_context_for_chat()

    system_prompt = f"""
You are the health-information support assistant for a Streamlit app named Stool Image Review.
Answer in {support_language}.
Scope:
- Help users understand their stool-image classification result and describe possible causes, related risks, and basic self-care.
- Users may describe symptoms such as pain, fever, diarrhea, blood, mucus, color changes, duration, diet, medicines, or dehydration signs.
- Use the model result context when available, but do not treat it as a diagnosis.
- If no analysis result is available yet, explain that you can still discuss symptoms generally and suggest analyzing an image for more context.
- Explain that model results are educational and not a medical diagnosis.
- Mention urgent warning signs when relevant: heavy bleeding, severe abdominal pain, high fever, persistent abnormal stool, marked weakness, weight loss, or jaundice.
Rules:
- Do not diagnose disease.
- Do not provide personalized medical treatment.
- Keep answers concise, practical, and calm.
Current model-analysis context:
{result_context}
"""

    try:
        client = OpenAI(api_key=api_key)
        response = client.responses.create(
            model=os.getenv("OPENAI_MODEL", "gpt-5-nano"),
            instructions=system_prompt,
            input=conversation,
            max_output_tokens=350,
        )
        return response.output_text.strip()
    except Exception as exc:
        st.warning(f"{T['support_api_error']}: {exc}")
        return None


def get_analysis_context_for_chat():
    result = st.session_state.get("last_result")
    if not result:
        return "No image analysis result is available yet."

    pred_class = result.get("predicted_class", "Unknown")
    confidence = result.get("confidence_score", 0.0)
    probabilities = result.get("probability_table")
    details = class_text(pred_class) if pred_class in CLASS_DETAILS else {}

    top_rows = []
    if probabilities is not None:
        for row in probabilities.head(3).itertuples(index=False):
            top_rows.append(f"{row.Class}: {row.Probability:.1f}%")

    return "\n".join(
        [
            f"Predicted class: {pred_class}",
            f"Localized label: {details.get('label', pred_class)}",
            f"Confidence: {confidence:.1%}",
            f"Top probabilities: {', '.join(top_rows) if top_rows else 'Not available'}",
            f"Summary: {details.get('summary', 'Not available')}",
            f"Possible causes: {'; '.join(details.get('causes', []))}",
            f"Possible risks: {'; '.join(details.get('risks', []))}",
            f"Basic care: {'; '.join(details.get('care', []))}",
        ]
    )


def add_support_message(role: str, content: str):
    st.session_state.support_messages.append({"role": role, "content": content})


def ensure_support_state():
    if st.session_state.get("support_lang") != LANG:
        st.session_state.support_lang = LANG
        st.session_state.pop("support_messages", None)

    if "support_messages" not in st.session_state:
        st.session_state.support_messages = []


@st.dialog("Ask OpenAI")
def render_support_dialog():
    ensure_support_state()
    api_ready = bool(get_openai_api_key())

    st.markdown(
        f"""
        <div>
            <h3 style="margin-top:0; margin-bottom:.35rem;">{T["support_title"]}</h3>
            <p style="color:#667784; margin-top:0;">{T["support_intro"]}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if not api_ready:
        st.info(T["support_api_missing"])

    for message in st.session_state.support_messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    st.markdown(f'<div class="support-note">{T["support_disclaimer"]}</div>', unsafe_allow_html=True)
    st.caption(T["support_quick"])
    quick_cols = st.columns(3)
    for index, question in enumerate([T["quick_blur"], T["quick_result"], T["quick_medical"]]):
        with quick_cols[index]:
            if st.button(question, key=f"quick_support_{LANG}_{index}", use_container_width=True):
                add_support_message("user", question)
                if api_ready:
                    reply = build_support_reply(question)
                    if reply:
                        add_support_message("assistant", reply)
                st.rerun()

    if st.button(T["support_reset"], use_container_width=True):
        st.session_state.support_messages = []
        st.rerun()

    user_message = st.chat_input(T["support_input"])
    if user_message:
        add_support_message("user", user_message)
        if api_ready:
            reply = build_support_reply(user_message)
            if reply:
                add_support_message("assistant", reply)
        st.rerun()


st.markdown(
    f"""
    <div class="hero">
        <div>
            <h1>{T["hero_title"]}</h1>
            <p>{T["hero_text"]}</p>
        </div>
        <div class="status-pill">{T["status"]}</div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(f'<div class="notice">{T["notice"]}</div>', unsafe_allow_html=True)

if st.button(T["open_chat"], type="secondary", key="open_chat_fab"):
    render_support_dialog()

with st.sidebar:
    st.header(T["control"])
    mode = st.radio(T["source"], [T["sample_mode"], T["upload_mode"]], index=0)
    blur_image = st.toggle(T["blur"], value=True)
    st.caption(T["sidebar_note"])


try:
    with st.spinner(T["load_model"]):
        learner = load_model(MODEL_PATH)
except Exception as exc:
    st.error(f"{T['model_error']}: {exc}")
    st.stop()


left_col, right_col = st.columns([0.9, 1.1], gap="large")
selected_image = None
selected_caption = ""

with left_col:
    st.subheader(T["choose_image"])

    if mode == T["sample_mode"]:
        samples = list_sample_images()
        if not samples:
            st.warning(T["no_samples"])
        else:
            class_name = st.selectbox(
                T["choose_class"],
                list(samples.keys()),
                format_func=lambda key: f"{class_text(key)['label']} ({key})",
            )
            sample_paths = samples[class_name]
            sample_options = {
                f"{index}.{class_name}": path
                for index, path in enumerate(sample_paths, start=1)
            }
            sample_label = st.selectbox(T["choose_sample"], list(sample_options.keys()))
            selected_image = sample_options[sample_label]
            selected_caption = sample_label
    else:
        selected_image = st.file_uploader(
            T["upload_file"],
            type=["jpg", "jpeg", "png", "webp"],
            accept_multiple_files=False,
        )
        selected_caption = T["uploaded_caption"] if selected_image else ""

    source_id = get_image_source_id(selected_image)
    clear_result_if_changed(source_id)

    if selected_image:
        render_preview_image(selected_image, selected_caption, blur_image)
    else:
        st.info(T["select_to_start"])

with right_col:
    st.subheader(T["result_title"])

    current_result = st.session_state.get("last_result")
    current_result_matches = current_result and current_result.get("source_id") == source_id

    if not selected_image:
        st.info(T["no_image"])
    else:
        if st.button(T["analyze"], type="primary", use_container_width=True):
            with st.spinner(T["analyzing"]):
                predicted_class, confidence_score, probability_table = predict_image(learner, selected_image)
            st.session_state.last_result = {
                "source_id": source_id,
                "predicted_class": predicted_class,
                "confidence_score": confidence_score,
                "probability_table": probability_table,
            }
            current_result = st.session_state.last_result
            current_result_matches = True

        if current_result_matches:
            render_result(
                current_result["predicted_class"],
                current_result["confidence_score"],
                current_result["probability_table"],
            )
        else:
            st.info(T["press_analyze"])

st.caption("Ai Builder Season 5")
