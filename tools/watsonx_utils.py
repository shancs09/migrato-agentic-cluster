
from dotenv import load_dotenv
import re,os,json,numpy as np
from ibm_watsonx_ai import Credentials
from ibm_watsonx_ai.foundation_models import ModelInference,Embeddings
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams,EmbedTextParamsMetaNames as EmbedParams
from ibm_watsonx_ai.foundation_models.utils.enums import ModelTypes, DecodingMethods,EmbeddingTypes


# Load environment variables from .env file (if using dotenv for environment variables)
load_dotenv()

wx_api_key = os.getenv('wx_api_key')
wx_service_url = os.getenv('wx_service_url')
wx_project_id = os.getenv('wx_project_id')
wx_llm_model_id = os.getenv('wx_llm_model_id', 'mistralai/mistral-medium-2505')  # Default value in case ENV is missing
wx_embedding_model=os.getenv('wx_embedding_model','ibm/slate-125m-english-rtrvr')

# To display example params enter # print(GenParams().get_example_values())
generate_params = {
    GenParams.DECODING_METHOD:'greedy',
    GenParams.MAX_NEW_TOKENS: 250,
    GenParams.STOP_SEQUENCES:["}\n"]
}

model_inference = ModelInference(
    # model_id=ModelTypes.GRANITE_20B_MULTILINGUAL,
    model_id=wx_llm_model_id, # Using utils: ModelTypes.MIXTRAL_8X7B_INSTRUCT_V01
    credentials=Credentials(
        api_key = wx_api_key,
        url = wx_service_url),
        project_id=wx_project_id
    )

# Embedding Params:
embed_params = {
    EmbedParams.TRUNCATE_INPUT_TOKENS: 512,
    EmbedParams.RETURN_OPTIONS: {"input_text": False}
}

wx_embeddings = Embeddings(
    model_id=wx_embedding_model,
    params=embed_params,
    credentials=Credentials(api_key=wx_api_key, url=wx_service_url),
    project_id=wx_project_id
)

def inference_llm_dutch(context_passages):
    llm_instr = """
            <s>[INST] <<SYS>>
            Je bent een uiterst capabele AI-assistent die documenten in meerdere talen intelligent classificeert. In deze taak analyseer je de verstrekte inhoud van een document en bepaal je het **meest geschikte enkele label** dat het type document het best beschrijft.

            ### Taal van het Antwoord:
            - **Ga ervan uit dat de documentinhoud in het Nederlands is.**
            - Geef het volledige antwoord in het **Nederlands**.

            ### Classificatierichtlijnen:
            - Analyseer de volledige documentinhoud zorgvuldig, rekening houdend met zowel structuur als semantiek.
            - Identificeer het **meest passende label** dat het doel van het document het beste samenvat.
            - **Geef de voorkeur aan specificiteit:** Als de inhoud overeenkomt met veelvoorkomende zakelijke documenttypen, geef dan de voorkeur aan het *meest specifieke en gedetailleerde* label uit uw kennisbank. Als een zeer specifiek label niet geschikt is, overweeg dan bredere categorieën zoals (gepast vertaald):
                "Factuur", "Notulen", "Agenda", "Interne Memo", "HR-beleidsdocument", "Technische Specificatie", "Productcatalogus", "Juridisch Contract", "Projectplanning", enz.
                Als een document bijvoorbeeld een "Service Level Agreement" is, geef dan de voorkeur aan "Service Level Agreement" boven "Juridisch Contract" als die mate van specificiteit door de inhoud wordt ondersteund.
            - Als het document hierboven niet duidelijk overeenkomt, gebruik dan uw beste oordeel om het **meest specifieke, geschikte en betekenisvolle label** te creëren dat mogelijk is op basis van de inhoud. Dit gecreëerde label moet het primaire doel en de inhoud van het document zo nauwkeurig mogelijk weergeven als uw verklaring.
            - **Consistentiecontrole:** Het gekozen 'label' moet de kern van uw 'verklaring' direct en beknopt samenvatten.
            - Het 'label' moet bestaan uit **maximaal 3 woorden** (bij voorkeur één kernwoord of korte frase).
            - Geef extra context of nuance alleen in de 'explanation', niet in het 'label'.


            ### Antwoordformaat (Moet geldige JSON zijn in dezelfde taal als het document):
            ```json
            {{
            "label": "<Meest geschikte enkele label voor het document>",
            "explanation": "<Korte uitleg waarom dit label is gekozen op basis van de inhoud. Het 'label' moet een beknopte en directe samenvatting zijn van deze uitleg.>"
            }}
            ```
            Voeg niets toe buiten deze JSON-structuur.
            <</SYS>>

            Document Summary:
            {doc_snippet}

            Wat is het meest geschikte label voor dit document? [/INST] """

    formatted_prompt = llm_instr.format(doc_snippet=context_passages)
    generated_response = model_inference.generate(prompt=formatted_prompt, params=generate_params)
    llm_response = generated_response['results'][0]['generated_text']
    
    llm_json_response = extract_json(llm_response)
    print("llm_json_response")
    print(llm_json_response)
    return llm_json_response

def inference_llm(context_passages):
    llm_instr = """
    <s>[INST] <<SYS>>
   You are a highly capable AI assistant designed to intelligently classify documents in multiple languages. Your task is to analyze the provided multi-page document content and determine the **most suitable single label** that best describes the overall type of document.

    ### Response Language:
    - Detect the language of the provided content (e.g., English, German, French,Dutch).
    - Respond entirely in the **same language** as the document content.

    ### Classification Guidelines:
    - Carefully analyze the full document content, considering both structure and semantics.
    - Identify the **most appropriate label** that best summarizes the document’s purpose.
    - **Prioritize Specificity:** If the content matches common business document types, prioritize the *most specific and granular* label from your knowledge base. If a highly specific label isn't suitable, then consider broader categories such as (translated appropriately):
        "Invoice", "Meeting Minutes", "Agenda", "Internal Memo", "HR Policy Document", "Engineering Specification", "Product Catalog", "Legal Contract", "Project Timeline", "Research Paper", "Financial Report", "User Manual", etc.
        For example, if a document is a "Service Level Agreement", prefer "Service Level Agreement" over "Legal Contract" if that level of specificity is supported by the content.
    - If the document doesn't clearly match the above, use your best judgment to **create the most specific, suitable, and meaningful label** possible based on the content. This created label should accurately reflect the document's primary purpose and content as closely as your explanation.
    - **Consistency Check:** The chosen 'label' must directly and concisely summarize the core idea presented in your 'explanation'.

    ### Response Format (Must be valid JSON in same language as document):
    ```json
    {{
      "label": "<Best single label for the document>",
      "explanation": "<Brief explanation of why this label was chosen based on the content. The 'label' should be a concise and direct summary of this explanation.>"
    }}
    ```
    Do not add anything outside this JSON structure.
    <</SYS>>

    Document Summary:
    {doc_snippet}

    What is the most suitable label for this document?
    [/INST]
    """

    formatted_prompt = llm_instr.format(doc_snippet=context_passages)
    generated_response = model_inference.generate(prompt=formatted_prompt, params=generate_params)
    llm_response = generated_response['results'][0]['generated_text']
    llm_json_response = extract_json(llm_response)
    return llm_json_response

def extract_json(output_text):
    """Extract JSON object from LLM output."""
    json_match = re.search(r'\{.*?\}|\[.*?\]', output_text, re.DOTALL)
    
    if json_match:
        try:
            json_object = json.loads(json_match.group(0))
            if isinstance(json_object, list) and len(json_object) > 0:
                return json_object[0]  # Return first element if list
            return json_object  # Otherwise, return dictionary
        except json.JSONDecodeError:
            return {"error": "Invalid JSON format", "original_output": output_text}
    else:
        return {"error": "No JSON object found", "original_output": output_text}  



# Run the application
if __name__ == "__main__":    

    llm_context=f"Filename: Microsoft 365 - SOC 1 Type 1 Report  (05-15-2024).pdf\n\nenvironment.  \nData Classification \nDefinition \nAccess Control Data \nData used to manage access to administrative roles or sensitive functions. \nCustomer Content \nContent directly created by users. Content is not viewed by Microsoft personnel \nunless required to resolve a ticketed service problem. \nEnd User Identifiable \nInformation (EUII) \nData unique to a user or generated from a user’s use of the service:  \n• Linkable to an individual user  \n• Does not contain customer content\n\nFilename: Azure - IRIS ICCS & ICP SOC 2 Type 1 Report (2018).pdf.pdf\n\nencryption key is restricted to authorized individuals. \nCCL-72 – Services Production environment uses different encryption keys than those for the Pre-Production \nenvironment (PPE).  \nCCL-98 – The Production and Pre-Production environment (PPE) are separated. New features and major \nchanges are developed and tested in separate environments prior to production implementation. Production \ndata is not replicated in test or development environments.\n\nFilename: Microsoft General - NGP PIMS SSAE 18 SOC 2 Report.pdf\n\nService auditors’ report for Microsoft NGP-PIMS - Page 11 \nData  \nData is maintained in Azure services and server databases. Each service team and support team is \nresponsible for managing security and availability of the data on the database servers. Reference the table \nbelow for the defined data classifications for this report and the NGP-PIMS environment.  \nData classification \nDefinition \nAccess control data \nData used to manage access to administrative roles or sensitive functions"
    