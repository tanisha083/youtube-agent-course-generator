import React from "react";
import { BrowserRouter as Router, Routes, Route } from "react-router-dom";
import UploadPage from "./pages/UploadPage";

const App: React.FC = () => {
    return (
        <Router>
            <Routes>
                {/* <Route path="/" element={<HomePage />} /> */}
                <Route path="/" element={<UploadPage />} />
            </Routes>
        </Router>
    );
};

export default App;
