import streamlit as st
from transformers import pipeline
import torch
import os, json, re
import tempfile

# write access token in secrets or .bashrc
token = os.environ.get('HF_TOKEN')

model_id = "google/gemma-3n-e2b-it"
image_types = ["jpg", "jpeg", "png", "bmp", "webp", "tiff"]

def input_prompt1(selected_goals_str):
    data = {
        "meal_title": "",
        "food_items": [
            "item 1",
            "item 2"
        ],
        "nutrition_info": {
            "calories_kcal": "0",
            "protein_g": "0",
            "carbs_g": "0",
            "fat_g": "0"
        },
        "comments": ""
    }
    json_string = json.dumps(data)
    return f"""
You are a highly specialized nutrition analysis and dietetics assistant. 
The goals of your client include {selected_goals_str}. 
Your sole purpose is to analyze meal images and provide structured nutritional information and comments based on all the goals of your client.
 
**Core Task & Output Format:** 

1. **Analyze Individual Meals:** 
* For each image provided, give the meal a title, identify all food items, estimate their quantity, and calculate a nutritional breakdown. 
* Output format MUST be a single JSON object per image in this form:
```json
{json_string} 
```
* The JSON object MUST have EXACTLY these four keys: 
* `meal_title`: A concise, descriptive title for the meal.
* `food_items`: A list of strings, where each string is a detected food item.
* `nutrition_info`: A dictionary containing estimated nutritional values. This MUST include keys for `calories_kcal`, `protein_g`, `carbs_g`, and `fat_g`. Provide reasonable estimations in a single number.
* `comments`: A string providing in-depth, helpful and non-judgmental comments and recommendations on the meal's nutritional value based on all the goals of your client. Comment explicitly on all the goals and give recommendations if applicable.

2. **Handle Non-Food Images:** 
* If an image does NOT contain food items, you MUST completely ignore it and provide NO output for that image. Do not generate any text, JSON, or even a comment.

3. **Output ONLY the JSON object(s):**
* Do not include any introductory or conversational text like "Here is the analysis," or "I've analyzed the images."
"""

def input_prompt2(json_list):
    return f"""
You are a highly specialized nutrition analysis and dietetics assistant. 
The goals of your client include {selected_goals_str}.
Here is the list of foods and their nutrition info and comments your client had in a day:
{json_list}
In less than 150 words, provide a single-paragraph summary of the OVERALL daily nutritional balance based on all the goals of your client.
Do NOT comment on any individual meal. 
Comment explicitly on all the goals of your client and give recommendations if applicable.
Do NOT provide numbers. Do NOT include a disclaimer.
"""

nutrition_goals = [
    "Weight Loss",
    "Weight Gain",
    "Manage Diabetes",
    "Lower Cholesterol",
    "Improve Digestion",
    "Increase Energy",
    "Mindful Eating",
    "Eat a Balanced Diet",
    "Plant-Based Diet",
    "Gluten-Free Diet",
    "Support Pregnancy",   
    "Fuel Athletic Performance",
]

@st.cache_resource
def model_setup(model_id):
    pipe = pipeline(
    "image-text-to-text",
    model=model_id,
    device="mps",
    torch_dtype=torch.bfloat16,
    use_auth_token=token,
    )
    return pipe

def run_model(prompt,image=[]):
    # Construct the input 
    content = []

    # Add text
    content.append({"type": "text", "text": 'Strictly follow the system instructions.'})

    # Add image
    for img in image:
        content.append({"type": "image", "image": img})

    messages = [
    {
        "role": "system",
        "content": [{"type": "text", "text": prompt}]
    },
    {
        "role": "user",
        "content": content
    }
    ]

    # Generate output
    output = pipe(text=messages, max_new_tokens=2000) #, temperature=0.2, verbose=False)
    return output[0]["generated_text"][-1]["content"]

def initialize():
    """
    Initializes chat history
    """
    st.session_state.messages = []

def extract_json_from_string(text):
    """
    Extracts all JSON objects contained within '```json' and '```' code blocks from a string.

    Args:
        text (str): The input string, which may contain text and one or more
                    JSON code blocks.

    Returns:
        list: A list of Python dictionaries, where each dictionary represents
              a successfully parsed JSON object from a code block. Returns an
              empty list if no valid JSON is found.
    """
    json_objects = []
    
    # The regex pattern looks for '```json' followed by any characters (.*?)
    # until it finds '```'. re.DOTALL allows the '.' to match newline characters,
    # which is essential for multi-line JSON. The '?' makes the matching non-greedy.
    pattern = re.compile(r'```json(.*?)```', re.DOTALL)
    
    matches = pattern.findall(text)
    
    for match in matches:
        # The match variable contains the content *between* the delimiters.
        json_string = match.strip()  # Clean up any leading/trailing whitespace
        
        if not json_string:
            continue
            
        try:
            # Attempt to parse the cleaned string as JSON
            json_obj = json.loads(json_string)
            json_objects.append(json_obj)
        except json.JSONDecodeError as e:
            print(f"Warning: Failed to decode JSON from a code block. Error: {e}")
            print(f"Mal-formed JSON content:\n{json_string}\n{'-'*20}")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            
    return json_objects

def process_numbers_from_string(text):
    """
    Extracts positive numbers from a string and performs an action based on the count.

    Args:
        text (str): The input string to search for numbers.

    Returns:
        float or None:
            - The single positive number found, as a float.
            - The mean of two positive numbers found, as a float.
            - None if zero or more than two numbers are found.
    """
    # Regex pattern to find integers and floating-point numbers
    number_strings = re.findall(r'\d+\.?\d*', text)

    # Convert the list of number strings to a list of floats
    numbers = [float(n) for n in number_strings]

    if len(numbers) == 1:
        # For one number, return the number itself
        return int(numbers[0])
    elif len(numbers) == 2:
        # For two numbers, calculate and return their mean
        return int(sum(numbers) // 2)
    else:
        # For any other count (0 or > 2), return None
        print(f"Warning: Found {len(numbers)} numbers. Returning None.")
        return None
    
### Load the model
pipe = model_setup(model_id)

### Title
st.title("What do you eat in a day?")
st.subheader("Nutrition analysis for your need")
st.badge("Your nutrition goals")
st.write("Please select all that apply:")

# Create a list of columns to hold the checkboxes
num_columns = 4
columns = st.columns(num_columns)

# Create an empty dictionary to store the state of each checkbox
selected_goals = {}

# Loop through the goals and place each checkbox in a column
for i, goal in enumerate(nutrition_goals):
    # Use the modulus operator (%) to cycle through the columns
    with columns[i % num_columns]:
        # The goal text works well as a key here
        is_checked = st.checkbox(goal, key=goal)
        selected_goals[goal] = is_checked

selected_goals_str = ""
if any(selected_goals.values()):
    # Get a list of the goals that were checked
    goals_list = [goal for goal, is_checked in selected_goals.items() if is_checked]
    selected_goals_str = ", ".join(goals_list)

### upload images
st.badge("Your meals in a day")
uploaded_files = st.file_uploader("Upload the image(s) of all your meals", type=image_types, accept_multiple_files=True, on_change=initialize)
if uploaded_files:
    image_names = []
    images = []
    if not isinstance(uploaded_files, list):
        uploaded_files = [uploaded_files]

    for uploaded_file in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            image_names.append(uploaded_file.name)
            tmp_file.write(uploaded_file.read())
            images.append(tmp_file.name)

    # Display the images in columns for a nice gallery view
    num_columns = 4
    columns = st.columns(num_columns)

    for i, image in enumerate(images):
        # Use the modulus operator (%) to cycle through the columns
        with columns[i % num_columns]:
            st.image(image, caption=image_names[i], use_container_width=True)

    # Run the multimodal model
    # print('++++++++++++++++++++++++')
    # print(input_prompt1(selected_goals_str))
    # print('++++++++++++++++++++++++')
    response1 = run_model(input_prompt1(selected_goals_str), images)
    # print(response1)
    # print('++++++++++++++++++++++++')
    json_obj = extract_json_from_string(response1)
    # print(json_obj)
    # print('========================')
    # print(input_prompt2(json_obj))
    # print('++++++++++++++++++++++++')
    response2 = run_model(input_prompt2(json_obj))
    # print(response2)

    # Genenrate a response to the user
    list_meals = [obj['meal_title'] for obj in json_obj]
    list_foods = list(set([food for obj in json_obj for food in obj['food_items']]))
    v = [process_numbers_from_string(str(obj['nutrition_info'][nutri])) for obj in json_obj for nutri in obj['nutrition_info']]
    list_nutritions = [sum(v[i::4]) for i in range(4)]
    list_comments = [obj['comments'] for obj in json_obj]

    st.badge("Comments", color='violet')
    st.write(response2)
    for i,meal in enumerate(list_meals):
        st.write(f"Meal {i+1}: {meal}")
        st.write(list_comments[i])
    
    st.badge("Nutrition", color='violet')
    st.write(f"""
        calories: {list_nutritions[0]}kcal\n
        protein: {list_nutritions[1]}g\n
        carbs: {list_nutritions[2]}g\n
        fat: {list_nutritions[3]}g
    """)

    st.markdown(''.join([f":orange-badge[{food}] " for food in list_foods]))