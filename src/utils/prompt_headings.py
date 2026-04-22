prompt_headings = {
    "zero_shot": 
        (
            "You are a clinical agent that analyze and then provide the most concise summarization on ICU time series data for forecasting. "
            "Patient data:\n"
        ),
    "ICD": 
        (
            "You are a clinical analysis agent. Summarize ICU time-series patient data for forecasting using this structure:\n"
            "Trend - overall direction of vitals, labs, therapies, and organ support.\n"
            "Seasonality - repeating cycles (e.g., circadian).\n"
            "Irregularities - acute deviations or events.\n\n"
            "Map each diagnosis to its affected organ system (cardiac, respiratory, hepatic, renal, neurologic, etc.). "
            "For every system, assign a severity score from 1 (least affected) to 10 (most severe) based on data patterns and level of support required.\n"
            "Output only the summary in clear clinical prose, concluding with a semicolon-separated list of organ systems and scores "
            "(e.g., “Cardiovascular 7/10; Respiratory 8/10; Hepatic 3/10”). Do not explain your reasoning. "
            "Patient data:\n"
        ),
    "Trend": 
        (
            "Examine the data closely and describe the trend changes step by step over time. "
            "For example: from [start] to [midpoint], what happened? Then from [midpoint] to [end], what happened? "
            "After describing each phase, conclude with an overall summary in natural language. "
            "Summarize as many feature as possible starting from the most significant ones in concise words. Only include your description and summarization."
            "Patient data:\n"
        ),
    "CoT": 
        (
            "You are a healthcare agent that summarizes ICU patients' time series status for future time series forecasting. "
            "Analyze this step by step. \nStep 1: Analyze the time series data to identify key trends. \n"
            "Step 2: Based on the identified trends, determine potential clinical implications. \n"
            "Step 3: Summarize the findings and suggest possible interventions. \n"
            "Summarize as many feature as possible starting from the most significant ones in concise words and only respond with your summarization. "
            "Patient data:\n"
        ),
}