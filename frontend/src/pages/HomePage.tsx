import React from "react";
import { Link } from "react-router-dom";

const HomePage: React.FC = () => {
    return (
        <div style={{ textAlign: "center", marginTop: "2rem" }}>
            <h1>Welcome to YouTube Agent Course Generator</h1>
            <p>Transform YouTube transcripts into interactive learning modules.</p>
            <Link to="/upload" style={{ textDecoration: "none", color: "blue" }}>
                Go to Upload Page
            </Link>
        </div>
    );
};

export default HomePage;
