from pydantic import BaseModel, Field


class ChunkModel(BaseModel):
    chunk_id: str = Field(..., description="Unique deterministic identifier for the chunk")
    file_path: str = Field(..., description="Relative file path of the source markdown file")
    source_lib: str = Field(..., description="Target library name (e.g., fastapi, docker, pytorch)")
    section_title: str = Field(..., description="Title derived from filename or primary header")
    chunk_index: int = Field(..., description="Sequential index of the chunk within the document")
    content: str = Field(..., description="The chunk text payload")
    char_count: int = Field(..., description="Total character count of the chunk content")


class GroundTruthQA(BaseModel):
    qa_id: str = Field(..., description="Unique identifier for the benchmark QA pair")
    source_chunk_id: str = Field(..., description="The exact chunk_id containing the ground truth")
    source_lib: str = Field(..., description="The library name (e.g. fastapi, docker, pytorch)")
    file_path: str = Field(..., description="Relative file path of the source markdown file")
    question: str = Field(..., description="Generated technical developer question")
    ground_truth_answer: str = Field(..., description="Concise, factual answer extracted from the chunk")

class QAGenerationResponse(BaseModel):
    question: str = Field(..., description="A realistic technical question asked by a developer")
    ground_truth_answer: str = Field(..., description="Factual answer based only on the provided snippet")
