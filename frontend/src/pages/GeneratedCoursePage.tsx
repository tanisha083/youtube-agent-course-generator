// src/pages/GeneratedCoursePage.tsx
import React from "react";
import { useLocation, useNavigate } from "react-router-dom";

interface LocationState {
  courseContent: string;
}

const GeneratedCoursePage: React.FC = () => {
  const location = useLocation();
  const navigate = useNavigate();
  const state = location.state as LocationState;
  const courseContent = state?.courseContent || "No course content available.";

  return (
    <div style={{ padding: "2rem", maxWidth: "800px", margin: "0 auto" }}>
      <h1>Generated Course</h1>
      <div style={{ marginTop: "1rem", whiteSpace: "pre-wrap" }}>
        {courseContent}
      </div>
      <button
        onClick={() => navigate("/")}
        style={{ marginTop: "1rem", padding: "0.5rem 1rem", cursor: "pointer" }}
      >
        Back to Home
      </button>
    </div>
  );
};

export default GeneratedCoursePage;
export {};