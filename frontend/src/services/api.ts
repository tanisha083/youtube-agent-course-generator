import axios, { AxiosResponse } from "axios"; 
import { ParsedBackendResponse } from "../pages/UploadPage"; 

const API_BASE_URL = "http://localhost:8000/api";

export const uploadTranscript = async (videoUrl: string): Promise<AxiosResponse<ParsedBackendResponse>> => {
    const response = await axios.post(`${API_BASE_URL}/generate-course`, { videoUrl });
    return response; 
};

export {}; 