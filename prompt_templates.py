# Unified prompt templates for all tasks
PROMPT_TEMPLATES = {
    "basic": [
        "Describe this image in detail.",
        "What do you see in this image?",
        "Analyze the contents of this image.",
        "Explain what's happening in this image.",
        "Provide a detailed description of this image."
    ],
    
    "analytical": [
        "Describe what you see in this image, considering its visual characteristics.",
        "Based on the image analysis, what object or subject is shown?",
        "How do the color and brightness patterns relate to the content?",
        "Analyze the image features and explain what they reveal about the subject.",
        "Describe how the regional brightness differences inform the image content.",
        "What patterns or structures can you identify from the edge density?",
        "How do the dominant colors help identify the subject?",
        "Explain how the brightness distribution relates to the image composition."
    ],
    
    "technical": [
        "Analyze the color distribution and its impact on the image content.",
        "How do the edge patterns contribute to the object recognition?",
        "Describe the relationship between brightness distribution and subject matter.",
        "What can you infer from the regional brightness variations?",
        "How do the dominant colors influence the perception of the subject?"
    ],
    
    "comparative": [
        "Compare the visual characteristics of this image with typical examples of its class.",
        "How does this image's color distribution differ from other images in its category?",
        "What makes this particular instance unique within its class?",
        "Analyze how this image's features align with or deviate from its class characteristics."
    ],

    "focused": [
        "What is the exact color and appearance of this {class_name}?",
        "What specific pose or position is this {class_name} in?",
        "What distinguishing features can you see clearly?",
        "How does this {class_name} stand out from its background?",
        "What makes this particular {class_name} unique or interesting?"
    ]
}

# Response templates for training
RESPONSE_TEMPLATES = {
    "basic": "This image shows a {class_name}. It is a clear and detailed view, showing distinctive features typical of a {class_name}.",
    "detailed": "This image depicts a {class_name}. The visual characteristics include {features}. The color distribution shows {color_info}, and the edge patterns suggest {edge_info}.",
    "analytical": "The image presents a {class_name}. Analysis of its visual properties reveals: {analysis}. These characteristics are consistent with typical {class_name} features.",
    "focused": "Looking at this {class_name}, I can see the following specific details: {details}"
}

# Image analysis templates
ANALYSIS_TEMPLATES = {
    "color": "The image has average RGB values of R:{r}, G:{g}, B:{b} with variations of ±{r_std}, ±{g_std}, ±{b_std}.",
    "brightness": "The overall brightness is {brightness}/255 with a contrast level of {contrast}.",
    "edges": "Edge density analysis shows {edge_density}% of pixels forming edges.",
    "regions": "Regional brightness varies with top: {top}, bottom: {bottom}, left: {left}, right: {right}."
}