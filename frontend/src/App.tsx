import React from "react";
import { BrowserRouter as Router, Routes, Route } from "react-router-dom";
import HomePage from "./pages/HomePage";
import UploadPage from "./pages/UploadPage";
import GeneratedCoursePage from "./pages/GeneratedCoursePage";

const App: React.FC = () => {
    return (
        <Router>
            <Routes>
                <Route path="/" element={<HomePage />} />
                <Route path="/upload" element={<UploadPage />} />
                <Route path="/generated-course" element={<GeneratedCoursePage />} />
            </Routes>
        </Router>
    );
};

export default App;
