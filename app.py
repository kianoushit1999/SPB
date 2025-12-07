from dotenv import load_dotenv
from agents import Agent, Runner, function_tool
import gradio as gr
from pydantic import BaseModel, Field
import math
import json
from openai import OpenAI
import base64
from io import BytesIO
from PIL import Image
from google import genai
from google.genai import types
from typing import Dict
import asyncio

load_dotenv(override=True)
client = genai.Client(api_key=GEMINI_API_KEY)

question_index = 0
repeat_question = False
accepted_answers = []

questionsList = [
        'سن شما چقدر است؟',
        'جنسیت شما چه میباشد؟(مرد / زن)',
        'آخرین فشار خون سیستولیک/دیاستولیک شما چند بوده است؟',
        'رژیم غذایی شما چقدر شور است؟ (کم / متوسط / زیاد)',
        'آیا به غذای خود نمک اضافه میکنید؟',
        'چند بار در هفته غذاهای پرنمک مصرف میکنید؟',
        'روزانه چند لیوان آب مینوشید؟',
        'آیا نوشابه یا قهوه زیاد مصرف میکنید؟',
        'چقدر ورزش میکنید ؟(هیچ / هفته ۱-۲ بار / منظم)',
        'قد فعلی شما چقدر است؟',
        'وزن فعلی شما چقدر است؟',
        'آیا فشار خون دارید؟',
        'چه دارویی مصرف میکنید',
        'آیا در خانواده تان کسی فشارخون یا سکته کرده است؟',
        'سطح استرس شما چقدر است؟',
        'آیا سیگار یا الکل مصرف میکنید؟',
    ]

def generate_user_info_instructions():
    global question_index, repeat_question

    return f'''
        You are an interviewer agent who is communicating in persian. Asking the question at once.

        Your goal is to find the answer of question.
        You must:
        1. Ask one question at a time
        2. Wait for the user's answer
        3. Validate if the answer is acceptable
        4. If invalid → ask that question again politely to get its related answer.
        5. If valid → store it and ask the next question


        You MUST NOT produce the final structured output until all fields are collected.
        Required questions (ask in Persian):
        {questionsList[question_index]}

        {f'You must say in more polite way that your previous answer is wrong and explain the {questionsList[question_index]} in more detail' if repeat_question else ''}

        - is_question_being_answered:
            true → only if the user's message is a valid answer to the current question. You need to checkout the response to be completely rational not to be everything.
            false → if the answer is invalid, irrelevant, empty, or incomplete.

        - accepted_user_response:
            If the answer is valid → return the raw user message.
            If invalid → return an empty string "".

        - agent_response:
            This is the text you want to send back to the user. Checkout the response to be rationally acceptable.
            If valid → say your answer being confirmed.
            If invalid → politely explain the mistake and ask the same question again.

        ** Crusial: if is_question_being_answered is true then 
    '''

class UserInfo(BaseModel):
    is_question_being_answered: bool = Field(description="Whether the question's answer is accpeted or not")
    accepted_user_response: str = Field(description="The user's response which was accepted")
    agent_response: str = Field(description="The response of the agent")

def call_interviewer_agent():
    agent_interviewer = Agent(
        name="Interviewer",
        instructions=generate_user_info_instructions(),
        model="gpt-4o",
        output_type = UserInfo
    )
    return agent_interviewer

class KBloodPressure(BaseModel):
    K: int = Field(description="calculated k")

K_INSTRUCTIONS = '''

- K is 5 for normal users (low risk) and 10 for users who are high-risk (e.g., history of high blood pressure, salt-sensitive, sedentary, smoker, family history, high salt diet).
- Only output 5 or 10.
- Use the answers to relevant questions to assess risk:
    - 'آخرین فشار خون سیستولیک/دیاستولیک شما چند بوده است؟'
    - 'آیا فشار خون دارید؟'
    - 'رژیم غذایی شما چقدر شور است؟'
    - 'چقدر ورزش میکنید ؟'
    - 'آیا سیگار یا الکل مصرف میکنید؟'
    - 'آیا در خانواده تان کسی فشارخون یا سکته کرده است؟'
- K is needed for calculating the effect of salt on blood pressure.
'''

k_agent = Agent(
    name = 'k-finder', 
    instructions = K_INSTRUCTIONS,
    model = 'gpt-4o',
    output_type = KBloodPressure
)

def clamp_to_range(value: float | int) -> int:
    """
    Force a number into the inclusive range [105, 180].
    Returns an *int* (you can change to float if you need decimals).
    """
    return max(105, min(180, int(value)))

@function_tool
def calculate_blood_pressure(k: int, SBP_current: int):

    ten_percent_more_salt_SBP_new = clamp_to_range(SBP_current + k*math.log(1.1))
    current_SBP_new = clamp_to_range(SBP_current + k*math.log(1))
    ten_percent_less_salt_SBP_new = clamp_to_range(SBP_current + k*math.log(0.9 ))

    return {
        'SBP_new': {
            'ten_percent_more_salt': ten_percent_more_salt_SBP_new,
            'current': current_SBP_new,
            'ten_percent_less_salt': ten_percent_less_salt_SBP_new
        }
    }

@function_tool
def predict_blood_pressure(SBP_current: int, SBP_new_ten_precent_more: int, SBP_new_current: int, SBP_new_ten_percent_less: int):
    '''
        Calculate the blood pressure after 30 days based on the:
        SBP(t) = SBP_base + ΔSBP × (1 - e^( -t / τ ))
    '''
    SBP_base = SBP_current
    delta_SBP = SBP_new_ten_precent_more - SBP_current
    tau = 10
    t = 30

    predicted_SBP_new_ten_precent_more = clamp_to_range(SBP_base + delta_SBP * (1 - pow(math.e, -t / tau)))

    delta_SBP = SBP_new_current - SBP_current
    predicted_SBP_new_current = clamp_to_range(SBP_base + delta_SBP * (1 - pow(math.e, -t / tau)))

    delta_SBP = SBP_new_ten_percent_less - SBP_current
    predicted_SBP_new_ten_percent_less = clamp_to_range(SBP_base + delta_SBP * (1 - pow(math.e, -t / tau)))

    return {
        'SBP_predicted': {
            'ten_percent_more_salt': predicted_SBP_new_ten_precent_more,
            'current': predicted_SBP_new_current,
            'ten_percent_less_salt': predicted_SBP_new_ten_percent_less
        }
    }

k_agent_tool = k_agent.as_tool(tool_name='k_agent_tool', tool_description='finding k for calculating the effect of salt on bool pressure')
tools = [k_agent_tool, calculate_blood_pressure, predict_blood_pressure]

def generate_BP_expert_instruction():
    return '''
        You are a Blood Pressure Expert AI. Your goal is to help estimate and predict a patient's systolic blood pressure (SBP) under different scenarios of salt intake over 30 days. 

        You have access to the following tools:

        1. "k_agent_tool": Use this tool to determine the sensitivity parameter K based on the patient's information. K can only be 5 (normal) or 10 (hypertensive or salt-sensitive). K is needed for calculating the effect of salt on blood pressure.

        2. "calculate_blood_pressure": Use this tool to calculate the patient's new SBP immediately after changes in salt intake. This tool takes K and the current SBP and provides SBP values for three scenarios:
        - Ten percent more salt
        - Current salt intake
        - Ten percent less salt

        3. "predict_blood_pressure": Use this tool to predict the SBP after 30 days for each scenario using a dynamic model:
        SBP(t) = SBP_base + ΔSBP × (1 - e^(-t / τ))

        Instructions:

        - Always start by using "k_agent_tool" to determine K from the patient's answers.
        - Then use "calculate_blood_pressure" with input parameter of K which is calculated in previous state and the patient's current SBP based on the answer of the question of 'آخرین فشار خون سیستولیک/دیاستولیک شما چند بوده است؟'.
        - Finally, use "predict_blood_pressure" to calculate the SBP after 30 days for each scenario. The inputs of predict_blood_pressure are the following:
            - SBP_current: based on the answer of question 'آخرین فشار خون سیستولیک/دیاستولیک شما چند بوده است؟'
            - SBP_new_ten_precent_more, SBP_new_current, SBP_new_ten_percent_less are the Immediate SBP which was calculated in predict_blood_pressure
        - Be precise and only use the tools for calculations; do not guess numeric values.
        - Output should clearly show in a json format and do not add further any words and sentences :
            - calculated K with the name of k in the output json
            - SBP_base which was used in all of the tools with SBP_base in the output json
            - Immediate SBP for each scenario based on function of calculate_blood_pressure (the name of output should not change at all and should be SBP_new )
            - Predicted SBP after 30 days for each scenario based on the output of predict_blood_pressure (the name of output should not change at all and must be SBP_predicted)

        Patient inputs you may receive include current SBP and any relevant health/salt-sensitivity information.

    '''

def call_blood_pressure_expert():
    blood_pressure_expert_agent = Agent(
        name = 'blood_pressure_expert',
        instructions= generate_BP_expert_instruction(),
        model="gpt-4o",
        tools = tools
    )

    return blood_pressure_expert_agent

async def call_SBP_expert_agent() -> Dict[str, str]:
    global questionsList, accepted_answers
    prompt = ''
    for question, ans in zip(questionsList, accepted_answers):
        prompt += f'{question}: {ans}\\n'

    prompt += """
        Please determine:
        1) K (5 for normal, 10 if patient has high blood pressure or is salt-sensitive).
        2) Immediate_SBP for three salt scenarios: 10% less, current, 10% more.
        3) Predicted SBP after 30 days for each scenario.
        Return results clearly labeled.
    """

    predictor_agent = call_blood_pressure_expert()
    result = await Runner.run(predictor_agent, prompt)
    raw_text = result.final_output
    print(raw_text)
    return json.loads(raw_text.replace('```json', '').replace('```', '').strip())

def generate_chart_image_generator_instruction(immediate_SBP, predicted_SBP):
    CHART_IMAGE_GENERATOR_INSTRUCTION = f'''
        You are a strict and highly precise chart-rendering agent. Your task is to generate a clean, accurate data-visualization image consisting of THREE SEPARATE LINE-CHART PANELS stacked vertically:

    1. Panel 1: "10% Less"
    2. Panel 2: "Current"
    3. Panel 3: "10% More"

    Each panel represents ONE scenario only.

    You MUST use ONLY the numerical values provided in two structured inputs:
    - immediate_SBP = {{ 'ten_percent_less': {immediate_SBP['ten_percent_less_salt']}, 'actual (current)': {immediate_SBP['current']}, 'ten_percent_more': {immediate_SBP['ten_percent_more_salt']} }}
    - predicted_SBP = {{ 'ten_percent_less': {predicted_SBP['ten_percent_less_salt']}, 'actual (current)': {predicted_SBP['current']}, 'ten_percent_more': {predicted_SBP['ten_percent_more_salt']} }}

    ------------------------------------------------------------
    OVERALL VISUAL DESIGN (applies to all 3 panels)
    ------------------------------------------------------------
    • All three charts must have the SAME:
    - Y-axis label: "SBP (mmHg)"
    - X-axis categories: "Day 0" (start), "Day 30" (end)
    - Horizontal scale range: Auto-scale but must include both values.
    • Style: Very clean, minimal, medical-grade professional.
    • Line: Smooth, thin, with circular markers at both data points.
    • Value labels MUST appear next to both data points.
    • Colors:
    - 10% Less → Green
    - Current → Blue
    - 10% More → Red
    • No extra gridlines besides major axis lines.

    ------------------------------------------------------------
    PANEL RULES (for each of the 3 diagrams)
    ------------------------------------------------------------

    ### PANEL 1 → "10% Less"  
    • Start at SBP = immediate_SBP.ten_percent_less  
    • End at SBP = predicted_SBP.ten_percent_less  
    • Draw a SINGLE line connecting Day 0 → Day 30.
    • Color: Green.
    • Title the panel: "10% Less SBP Scenario".

    ### PANEL 2 → "Current"  
    • Start at SBP = immediate_SBP.current  
    • End at SBP = predicted_SBP.current  
    • Draw a SINGLE line connecting Day 0 → Day 30.
    • Color: Blue.
    • Title the panel: "Current SBP Scenario".

    ### PANEL 3 → "10% More"
    • Start at SBP = immediate_SBP.ten_percent_more  
    • End at SBP = predicted_SBP.ten_percent_more  
    • Draw a SINGLE line connecting Day 0 → Day 30.
    • Color: Red.
    • Title the panel: "10% More SBP Scenario".

    ------------------------------------------------------------
    STRICT ACCURACY REQUIREMENTS
    ------------------------------------------------------------
    • You must plot EXACT given numerical values — no smoothing, no rounding.
    • The Y-axis must reflect the true distance between immediate vs. predicted.
    • If the SBP rises, line slopes upward; if it falls, slope downward; if equal, line is horizontal.
    • Do NOT merge scenarios into one chart. Each panel = one scenario only.
    • Panels must be visually parallel and uniformly sized.

    ------------------------------------------------------------
    OUTPUT REQUIREMENT
    ------------------------------------------------------------
    Produce ONE final image that contains all three panels aligned vertically in the order:
    1. 10% Less  
    2. Current  
    3. 10% More  

    It must look like a medical-grade report visualization: clean, aligned, precise.

    '''

    return CHART_IMAGE_GENERATOR_INSTRUCTION

def digital_twin_image_prompt_generator():
    global accepted_answers, question_list
    DIGITAL_TWIN_IMAGE_PROMPT_INSTRUCTIONS = '''
        You are the **Digital Twin Image-Prompt Generator Agent**.

        Your job:
        - Receive the user's interview answers (age, gender, height, weight, BMI, baseline SBP, salt intake, activity level, smoking/alcohol, stress, family history, medications, current_SBP).
        - Analyze all values and convert them into **digital-twin parameters**.
        - Then produce a **complete and final image-generation prompt** for an external image generator model.

        ### 1. Processing Rules
        - Calculate BMI = weight / (height in meters)^2
        - Assign a **risk color** for the human silhouette:
        - Green = low risk
        - Yellow = moderate risk
        - Red = high risk
        - Create three surrounding rings:
        1. Salt ring → intensity based on salt intake + added salt + salty food frequency
        2. Weight ring → intensity based on BMI category
        3. Activity ring → intensity based on physical activity level
        - Predict SBP trend (simple rules):
        - High salt or no activity → predicted SBP increases
        - Healthy habits → predicted SBP decreases or stays stable

        ### 2. Image Prompt Requirements
        You must output **only the final image prompt**, describing:

        - A human digital silhouette ("digital twin")
        - The silhouette's color showing the total cardiovascular risk
        - Three circular rings around the figure:
        - Salt ring
        - Weight ring
        - Activity ring
        - Each ring's brightness/thickness scales with the user's risk levels
        - Display text labels with:
        - Age
        - Gender
        - BMI (calculated)
        - Current SBP
        - Predicted SBP
        - Style:
        - Clean
        - Modern medical infographic
        - High contrast
        - Soft glow effects around rings

        ### 3. Output Format
        ALWAYS output only the **image prompt**, nothing else.

        The output is meant to be fed directly into an image generation model.

    '''
    digital_twin_image_prompt_agent = Agent(
        name='image_prompt_generator_agent',
        model='gpt-4o',
        instructions = DIGITAL_TWIN_IMAGE_PROMPT_INSTRUCTIONS,
    )

    return digital_twin_image_prompt_agent

async def gemini_image_generator_async(prompt):
    def sync_generate():
        response = client.models.generate_content(
            model="gemini-2.5-flash-image",
            contents=prompt,
            config=types.GenerateContentConfig(response_modalities=[types.Modality.IMAGE])
        )
        image_bytes = response.candidates[0].content.parts[0].inline_data.data
        return Image.open(BytesIO(image_bytes))
    return await asyncio.to_thread(sync_generate)

WATCH_FILM_MESSAGE = "برای اطلاعات بیشتر ویدیو را تماشا کنید."

async def chat(message, history):
    global question_index, repeat_question

    history = history or []

    agent_interviewer = call_interviewer_agent()
    result = await Runner.run(agent_interviewer, message)

    print(result.final_output)

    chart_image = None
    digital_twin_image = None

    if (result.final_output.is_question_being_answered):
        question_index += 1
        repeat_question = False
        accepted_answers.append(result.final_output.accepted_user_response)
        if question_index == len(questionsList):
            agent_reply = (
                        str(result.final_output.agent_response)
                        + '\\n'
                        + f'{WATCH_FILM_MESSAGE}'
                    )

            json_SBP = await call_SBP_expert_agent()
            print(json_SBP)
            immediate_SBP = json_SBP['SBP_new']
            predicted_SBP = json_SBP['SBP_predicted']
            print(immediate_SBP, predicted_SBP)

            chart_image_generator_agent = digital_twin_image_prompt_generator()
            prompt = 'These are the questions and answers of interview: \\n'
            for question, ans in zip(questionsList, accepted_answers):
                prompt += f'{question}: {ans}\\n'
            image_prompt = await Runner.run(chart_image_generator_agent, prompt)
            chart_image, digital_twin_image = await asyncio.gather(gemini_image_generator_async(generate_chart_image_generator_instruction(immediate_SBP, predicted_SBP)), gemini_image_generator_async(image_prompt.final_output))


        else:
            agent_reply = str(result.final_output.agent_response) + '\n' + f'{questionsList[question_index]}'
    else:
        repeat_question = True
        agent_reply = str(result.final_output.agent_response)

    history.append((message, agent_reply))

    video_url = ''
    if (question_index == len(questionsList)):
        video_url = 'https://www.aparat.com/video/video/embed/videohash/pl2Ww/vt/frame'

    video_output = f"""<div style="width:100%; height:400px;">
                            <iframe
                                style="width:100%; height:100%; border:0;"
                                src="{video_url}"
                                allowfullscreen>
                            </iframe>
                        </div>
                    """    

    return history, "", video_output, chart_image, digital_twin_image

def reset_all():
    global question_index, repeat_question, accepted_answers
    question_index = 0
    repeat_question = False
    accepted_answers = []

    return "", "", "", None, None

with gr.Blocks() as demo:
    gr.Markdown("<h1 style='text-align:center; direction:rtl;'>🧠 دستیار مصاحبه سلامت</h1>")

    with gr.Row():
        with gr.Column(scale=2):
            chatbot = gr.Chatbot(label="گفتگو", rtl=True)
            msg = gr.Textbox(label="پیام شما", placeholder="پیام خود را وارد کنید...", rtl=True)
            reset_btn = gr.Button("🔄 ریست")
    with gr.Row():
        chart_image = gr.Image(label="📊 نمودار فشار خون")
        video_output = gr.HTML(label="ویدیو")
        digital_twin_image = gr.Image(label="🧍♂️ تصویر دیجیتال توئین")
    
            

    # Message submit
    msg.submit(
        chat,
        [msg, chatbot],
        [chatbot, msg, video_output, chart_image, digital_twin_image]
    )

    # Reset button click
    reset_btn.click(
        reset_all,
        inputs=[],
        outputs=[chatbot, msg, video_output, chart_image, digital_twin_image]
    )

if __name__ == "__main__":
    demo.launch()
