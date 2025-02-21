import React, { useState, useRef, useEffect } from 'react';
import {
    Box,
    Tabs,
    Tab,
    Accordion,
    AccordionSummary,
    AccordionDetails,
    Typography,
    Grid,
    Card,
    CardContent,
    Button,
    CircularProgress,
    Snackbar,
    Alert,
    ThemeProvider,
    createTheme
} from "@mui/material";
import ExpandMoreIcon from '@mui/icons-material/ExpandMore';
import { uploadTranscript } from "../services/api";
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';

const REACT_APP_API_BASE_URL = 'http://localhost:8000';

interface Frame {
    frame_path: string;
    caption: string;
    info: string;
    timestamp: number;
}

interface MediaItem {
    frame_path: string;
    caption: string;
    info: string;
    timestamp: number;
}

interface StructuredContentSection {
    section_title: string;
    start_ts: number;
    end_ts: number;
    frames: Frame[];
    media: [];
}

interface StructuredContentModule {
    module_title: string;
    sections: StructuredContentSection[];
}

interface StructuredContent {
    modules: StructuredContentModule[];
    global_concepts: string[];
}

interface CourseContentSection {
    section_title: string;
    content: string;
    media: MediaItem[];
}

interface CourseContentModule {
    module_title: string;
    sections: CourseContentSection[];
}

interface CourseContent {
    modules: CourseContentModule[];
    global_concepts: string[];
}


interface QuestionItem {
    type: string;
    text: string;
    options?: string[];
    correct_answer: string;
    rationale?: string;
    review_timestamp?: string | number; 
    difficulty?: number;
}

interface QuizItem {
    module_title: string;
    section_title: string;
    questions: QuestionItem[];
}

interface QuizContent {
    quizzes: QuizItem[];
}

interface RetentionTip {
    type: string;
    description: string;
    executed_content: string;
}

interface RetentionSectionItem {
    section_title: string;
    retention_tips?: RetentionTip[];
    visual_summary_suggestion?: string;
}

interface RetentionModule {
    module_title: string;
    retention_tips: RetentionTip[];
    spaced_repetition_prompts: string[];
    scenario_examples: string[];
    section_retention?: RetentionSectionItem[];
}

interface RetentionPlanDetail {
    module_retention: RetentionModule[];
    overall_summary: string;
}

interface RetentionPlan {
    retention_plan: RetentionPlanDetail;
}


export interface ParsedBackendResponse {
    course: {
        structured_content: StructuredContent;
        quiz_content: QuizContent;
        retention_plan: RetentionPlan;
        course_content: CourseContent;
    };
    frames: Frame[];
    transcript_file: string;
}

const theme = createTheme({  // <--- Define the theme here
    palette: {
        primary: {
            main: '#333', // Dark gray for a sleek look
        },
        secondary: {
            main: '#555',
        },
        text: {
            primary: '#333', // Consistent dark gray for text
            secondary: '#555',
        },
        background: {
            default: '#fff', // White background
            paper: '#fff', // White for cards and paper elements
        },
        success: {
            main: '#2ecc71', // Keep success color
        },
        error: {
            main: '#e74c3c', // Keep error color
        },
    },
    typography: {
        fontFamily: [
            'Inter', // Modern, sans-serif font.  Very readable.
            '-apple-system',
            'BlinkMacSystemFont',
            '"Segoe UI"',
            'Roboto',
            '"Helvetica Neue"',
            'Arial',
            'sans-serif',
            '"Apple Color Emoji"',
            '"Segoe UI Emoji"',
            '"Segoe UI Symbol"',
        ].join(','),
        h3: {
            fontSize: '2.2rem',
            fontWeight: 600, // Semi-bold
        },
        h5: {
            fontSize: '1.5rem',
            fontWeight: 600,
        },
        h6: {
            fontSize: '1.2rem',
            fontWeight: 600,
        },
        subtitle1: {
            fontSize: '1rem',
            fontWeight: 500,
        },
        body2: {
            fontSize: '0.9rem',
        },
        button: {
            fontWeight: 500, // Slightly bolder buttons
        }
    },
    components: {
        MuiButton: {
            styleOverrides: {
                root: {
                    borderRadius: 8,
                },
            },
        },
        MuiAccordion: {
            styleOverrides: {
                root: {
                    '&:before': {
                        display: 'none',
                    },
                },
            },
        },
    },
});


const UploadPage: React.FC = () => {
    const [videoUrl, setVideoUrl] = useState("");
    const [response, setResponse] = useState<ParsedBackendResponse | null>(null);
    const [activeTab, setActiveTab] = useState(0);
    const [selectedAnswers, setSelectedAnswers] = useState<{ [questionKey: string]: string }>({});
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState("");
    const [embeddedVideoUrl, setEmbeddedVideoUrl] = useState<string | null>(null);
    // const [playerReady, setPlayerReady] = useState(false);  // Track if the player is ready
    const playerRef = useRef<HTMLIFrameElement>(null);

    const extractVideoId = (url: string): string | null => {
        const regex = /(?:https?:\/\/(?:www\.)?)?youtu(?:\.be\/|be\.com\/(?:watch\?(?:feature=youtu\.be\&)?v=|v\/|embed\/))([a-zA-Z0-9_-]+)/;
        const match = url.match(regex);
        return match ? match[1] : null;
    };

    useEffect(() => {
        if (videoUrl) {
            const videoId = extractVideoId(videoUrl);
            if (videoId) {
                setEmbeddedVideoUrl(`https://www.youtube.com/embed/${videoId}?enablejsapi=1`);
            }
        }
    }, [videoUrl]);

    const formatTimestamp = (timestamp: number): string => {
        const hours = Math.floor(timestamp / 3600)
          .toString()
          .padStart(2, "0");
        const minutes = Math.floor((timestamp % 3600) / 60)
          .toString()
          .padStart(2, "0");
        const seconds = Math.floor(timestamp % 60)
          .toString()
          .padStart(2, "0");
      
        return `${hours}:${minutes}:${seconds}`;
      };

    const handleSubmit = async () => {
        try {
            setLoading(true);
            const result = await uploadTranscript(videoUrl);

            const courseData = {
                structured_content: result.data.course.structured_content,
                quiz_content: result.data.course.quiz_content,
                retention_plan: result.data.course.retention_plan,
                course_content: result.data.course.course_content,
            };
            console.log("Raw Backend Response Data:", result.data);
            console.log("Processed Response Data:", courseData);

            setResponse(() => ({
                ...result.data,
                course: courseData,
            }));

            setError("");
        } catch (err: any) {
            setError(err.message || "Failed to generate course. Please try again.");
            console.error(err);
        } finally {
            setLoading(false);
        }
    };

    const handleQuizAnswer = (questionKey: string, answer: string) => {
        setSelectedAnswers(prev => ({
            ...prev,
            [questionKey]: answer
        }));
    };

    const isAnswerCorrect = (questionKey: string, question: QuestionItem) => {
        return selectedAnswers[questionKey] === question.correct_answer;
    };

    return (
        <ThemeProvider theme={theme}>
            <Box sx={{ maxWidth: 1200, margin: "0 auto", p: 3 }}>
                <Typography variant="h3" gutterBottom align="center" sx={{ fontWeight: "bold", color: (theme) => theme.palette.primary.main }}>
                    Video Course Generator
                </Typography>
                <style>{`
                    table {
                        border-collapse: collapse;
                        margin: 1rem 0;
                        width: 100%;
                    }
                    td, th {
                        border: 1px solid #ddd;
                        padding: 8px;
                        text-align: left;
                    }
                    th {
                        background-color: #f5f5f5;
                        font-weight: 600;
                    }
                    strong {
                        font-weight: 600;
                    }
                `}</style>
                <Box sx={{ display: "flex", gap: 2, mb: 4 }}>
                    <input
                        type="text"
                        placeholder="Enter YouTube Video URL"
                        value={videoUrl}
                        onChange={(e) => setVideoUrl(e.target.value)}
                        style={{ flex: 1, padding: "12px", borderRadius: "8px", border: "1px solid #ddd", fontSize: "16px" }}
                    />
                    <Button
                        variant="contained"
                        onClick={handleSubmit}
                        disabled={loading}
                        sx={{ px: 4, py: 1.5, borderRadius: "8px" }}
                    >
                        {loading ? <CircularProgress size={24} /> : "Generate Course"}
                    </Button>
                </Box>

                {embeddedVideoUrl && (
                    <Box sx={{ mt: 2, mb: 4 }}>
                        <Typography variant="h6">Embedded Video</Typography>
                        <iframe
                            ref={playerRef}
                            width="100%"
                            height="500"
                            src={`${embeddedVideoUrl}`}
                            title="YouTube video player"
                            frameBorder="0"
                            allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
                            allowFullScreen
                        ></iframe>
                    </Box>
                )}

                {error && (
                    <Snackbar open={!!error} autoHideDuration={6000} onClose={() => setError("")}>
                        <Alert severity="error" onClose={() => setError("")}>{error}</Alert>
                    </Snackbar>
                )}

                {response && (
                    <Box sx={{ mt: 4 }}>
                        <Tabs value={activeTab} onChange={(_, newValue) => setActiveTab(newValue)}>
                            <Tab label="Course Content" sx={{ textTransform: 'none' }} />
                            <Tab label="Knowledge Check" sx={{ textTransform: 'none' }} />
                            <Tab label="Retention & Review" sx={{ textTransform: 'none' }} />
                        </Tabs>


                        {/* Course Content Tab (Tab 0) */}
                        {activeTab === 0 && response?.course?.course_content?.modules && (
                            <Box sx={{ mt: 3 }}>
                                <Typography variant="h5" gutterBottom>Course Content</Typography>
                                {response.course.course_content.modules.map((module, moduleIndex) => (
                                    <Accordion key={moduleIndex} elevation={0} square>
                                        <AccordionSummary expandIcon={<ExpandMoreIcon />} >
                                            <Typography variant="h6" sx={{ fontWeight: 'bold' }}>{module.module_title}</Typography>
                                        </AccordionSummary>
                                        <AccordionDetails>
                                            {module.sections.map((section, sectionIndex) => (
                                                <Box key={sectionIndex} sx={{ mb: 3 }}>
                                                    <Typography variant="subtitle1" sx={{ fontWeight: 'bold' }}>{section.section_title}</Typography>
                                                    <ReactMarkdown remarkPlugins={[remarkGfm]} children={section.content} />

                                                    {section.media && section.media.length > 0 && (
                                                        <Box sx={{ mt: 2 }}>
                                                            <Typography variant="subtitle2" sx={{ fontWeight: "bold" }}>Multimedia:</Typography>
                                                            <Grid container spacing={2}>
                                                                {section.media.map((mediaItem, mediaIndex) => (
                                                                    <Grid item xs={6} md={4} lg={3} key={mediaIndex}>
                                                                        <Card elevation={0} sx={{ border: '1px solid #ddd' }}>
                                                                            {mediaItem.frame_path && (
                                                                                <Box>
                                                                                    <img
                                                                                        src={`${REACT_APP_API_BASE_URL}${mediaItem.frame_path}`}
                                                                                        alt={mediaItem.caption}  // Use caption as alt text
                                                                                        style={{ width: "100%", height: "100%", objectFit: "fill" }}
                                                                                    />
                                                                                    <Button size="small" color="primary" sx={{ textTransform: 'none' }}>
                                                                                        Seek to timestamp {formatTimestamp(mediaItem.timestamp)}
                                                                                    </Button>
                                                                                </Box>
                                                                            )}
                                                                            <CardContent>
                                                                                {/* Removed Chip - timestamp no longer in mediaItem, it's in frame_path */}
                                                                                <ReactMarkdown
                                                                                    remarkPlugins={[remarkGfm]}
                                                                                    components={{
                                                                                        p: ({ children }) => <Typography component="p" variant="caption" display="block">{children}</Typography>,
                                                                                    }}
                                                                                    children={mediaItem.caption}
                                                                                />
                                                                            </CardContent>
                                                                        </Card>
                                                                    </Grid>
                                                                ))}
                                                            </Grid>
                                                        </Box>
                                                    )}
                                                </Box>
                                            ))}
                                        </AccordionDetails>
                                    </Accordion>
                                ))}
                            </Box>
                        )}


                        {/* Quiz Tab (Tab 1) - Sleek Design */}
                        {activeTab === 1 && response?.course?.quiz_content?.quizzes && (
                            <Box sx={{ mt: 3, }}>
                                <Typography variant="h5" gutterBottom sx={{ fontWeight: 'bold', textAlign: 'center', mb: 4, color: 'text.primary' }}>
                                    Interactive Quiz
                                </Typography>
                                {response.course.quiz_content.quizzes.map((quiz, quizIndex) => (
                                    <Accordion key={quizIndex} elevation={0} square sx={{ borderBottom: '1px solid #ddd', '&:before': { display: 'none' } }}>
                                        <AccordionSummary expandIcon={<ExpandMoreIcon />} sx={{ '& .MuiAccordionSummary-content': { margin: '12px 0', } }}>
                                            <Typography variant="h6" sx={{ fontWeight: 'bold', color: 'text.primary' }}>
                                                {quiz.module_title} - {quiz.section_title}
                                            </Typography>
                                        </AccordionSummary>
                                        <AccordionDetails sx={{ padding: '0 24px 24px 24px' }}>
                                            {quiz.questions.map((question, questionIndex) => {
                                                const questionKey = `${quiz.module_title}-${quizIndex}-${questionIndex}`;
                                                const selectedAnswer = selectedAnswers[questionKey];
                                                const isAnswered = selectedAnswer !== undefined;
                                                const isCorrect = isAnswered && isAnswerCorrect(questionKey, question);

                                                return (
                                                    <Card key={questionKey} elevation={0} sx={{ mb: 3, border: '1px solid #ddd', borderRadius: '8px' }}>
                                                        <CardContent>
                                                            <Typography variant="subtitle1" gutterBottom sx={{ fontWeight: 'bold', color: 'text.secondary', mb: 1 }}>
                                                                Question {questionIndex + 1}
                                                            </Typography>
                                                            <Typography variant="h6" gutterBottom sx={{ fontWeight: 500, color: 'text.primary', mb: 2 }}>
                                                                {question.text}
                                                            </Typography>

                                                            {question.type === 'multiple_choice' && question.options && (
                                                                <Grid container spacing={2}>
                                                                    {question.options.map((option, optIndex) => (
                                                                        <Grid item xs={12} sm={6} key={optIndex}>
                                                                            <Button
                                                                                fullWidth
                                                                                variant="outlined"
                                                                                color={isAnswered ? (selectedAnswer === option ? (isCorrect ? 'success' : 'error') : 'primary') : 'primary'}
                                                                                onClick={() => handleQuizAnswer(questionKey, option)}
                                                                                sx={{
                                                                                    textTransform: 'none',
                                                                                    justifyContent: 'flex-start',
                                                                                    padding: '10px 16px',
                                                                                    borderRadius: '8px',
                                                                                    borderColor: isAnswered ? (selectedAnswer === option ? (isCorrect ? '#2ecc71' : '#e74c3c') : '#ddd') : '#ddd',
                                                                                    '&:hover': {
                                                                                        backgroundColor: isAnswered ? (selectedAnswer === option ? (isCorrect ? 'rgba(46, 204, 113, 0.1)' : 'rgba(231, 76, 60, 0.1)') : 'rgba(0, 0, 0, 0.04)') : 'rgba(0, 0, 0, 0.04)',
                                                                                        borderColor: isAnswered ? (selectedAnswer === option ? (isCorrect ? '#2ecc71' : '#e74c3c') : '#aaa') : '#aaa',
                                                                                    },
                                                                                }}
                                                                            >
                                                                                <span style={{ marginRight: '8px', fontWeight: 'bold' }}>{String.fromCharCode(65 + optIndex)}.</span>
                                                                                {option}
                                                                            </Button>
                                                                        </Grid>
                                                                    ))}
                                                                </Grid>
                                                            )}

                                                            {question.type === 'true_false' && (
                                                                <Grid container spacing={2}>
                                                                    {["true", "false"].map((option) => (
                                                                        <Grid item xs={6} key={option}>
                                                                            <Button
                                                                                fullWidth
                                                                                variant="outlined"
                                                                                color={isAnswered ? (selectedAnswer === option ? (isCorrect ? 'success' : 'error') : 'primary') : 'primary'}
                                                                                onClick={() => handleQuizAnswer(questionKey, option)}
                                                                                sx={{
                                                                                    textTransform: 'none',
                                                                                    justifyContent: 'center',
                                                                                    padding: '10px 16px',
                                                                                    borderRadius: '8px',
                                                                                    borderColor: isAnswered ? (selectedAnswer === option ? (isCorrect ? '#2ecc71' : '#e74c3c') : '#ddd') : '#ddd',

                                                                                    '&:hover': {
                                                                                        backgroundColor: isAnswered ? (selectedAnswer === option ? (isCorrect ? 'rgba(46, 204, 113, 0.1)' : 'rgba(231, 76, 60, 0.1)') : 'rgba(0, 0, 0, 0.04)') : 'rgba(0, 0, 0, 0.04)',
                                                                                        borderColor: isAnswered ? (selectedAnswer === option ? (isCorrect ? '#2ecc71' : '#e74c3c') : '#aaa') : '#aaa',
                                                                                    }
                                                                                }}
                                                                            >
                                                                                {option.charAt(0).toUpperCase() + option.slice(1)}
                                                                            </Button>
                                                                        </Grid>
                                                                    ))}
                                                                </Grid>
                                                            )}

                                                            {isAnswered && (
                                                                <Box sx={{ mt: 2, p: 1.5, borderRadius: '8px', backgroundColor: isCorrect ? 'rgba(46, 204, 113, 0.1)' : 'rgba(231, 76, 60, 0.1)' }}>
                                                                    <Typography sx={{ color: isCorrect ? '#2ecc71' : '#e74c3c', fontWeight: 'bold' }}>
                                                                        {isCorrect ? 'Correct! 🎉' : 'Incorrect.'}
                                                                    </Typography>
                                                                    {!isCorrect && (
                                                                        <Typography sx={{ mt: 0.5, color: 'text.secondary' }}>
                                                                            {question.rationale || ""}
                                                                        </Typography>
                                                                    )}

                                                                     {question.review_timestamp && (
                                                                        <Typography variant="caption" sx={{ display: 'block', fontStyle: 'italic', mt: 0.5, color: 'text.secondary' }}>
                                                                           Review: {question.review_timestamp}
                                                                        </Typography>
                                                                    )}
                                                                </Box>
                                                            )}
                                                        </CardContent>
                                                    </Card>
                                                );
                                            })}
                                        </AccordionDetails>
                                    </Accordion>
                                ))}
                            </Box>
                        )}


                        {/* Study Plan Tab (Tab 2) */}
                        {activeTab === 2 && response?.course?.retention_plan?.retention_plan?.module_retention && (
                            <Box sx={{ mt: 3 }}>
                                {/* <Typography variant="h5" gutterBottom>Retention Strategy</Typography> */}
                                {response.course.retention_plan.retention_plan.module_retention.map((moduleRetention, moduleIndex) => (
                                    <Box key={moduleIndex} sx={{ mb: 4 }}>
                                        <Typography variant="h6" gutterBottom>{moduleRetention.module_title}</Typography>
                                        <Box sx={{ mt: 1 }}>
                                            <Typography variant="body2" sx={{ fontWeight: "bold" }}>
                                                Retention Tips:
                                            </Typography>
                                            <ul>
                                                {moduleRetention.retention_tips.map((tip, tipIndex) => (
                                                    <li key={tipIndex}>
                                                        {/* <Typography variant="body1" sx={{ fontWeight: "bold" }}>{tip.type}:</Typography> */}
                                                        <Typography variant="body2" sx={{ fontWeight: "bold" }}>{tip.description}</Typography>
                                                        <ReactMarkdown 
                                                            remarkPlugins={[remarkGfm]}
                                                            components={{
                                                                p: ({children}) => <Typography variant="body2" component="div" sx={{ mt: 1 }}>{children}</Typography>
                                                            }}
                                                        >
                                                            {tip.executed_content}
                                                        </ReactMarkdown>
                                                    </li>
                                                ))}
                                            </ul>
                                        </Box>

                                        {moduleRetention.spaced_repetition_prompts && (
                                            <Box sx={{ mt: 2 }}>
                                                <Typography variant="subtitle2" sx={{ fontWeight: "bold" }}>Active Recall Questions</Typography>
                                                <ul>
                                                    {moduleRetention.spaced_repetition_prompts.map((prompt, promptIndex) => (
                                                        <li key={promptIndex}>{prompt}</li>
                                                    ))}
                                                </ul>
                                            </Box>
                                        )}

                                        {moduleRetention.scenario_examples && (
                                            <Box sx={{ mt: 2 }}>
                                                <Typography variant="subtitle2" sx={{ fontWeight: "bold" }}>Scenario Examples:</Typography>
                                                <ul>
                                                    {moduleRetention.scenario_examples.map((example, exampleIndex) => (
                                                        <li key={exampleIndex}>{example}</li>
                                                    ))}
                                                </ul>
                                            </Box>
                                        )}
                                    </Box>
                                ))}
                                {response.course.retention_plan.retention_plan.overall_summary && (
                                    <Box sx={{ mt: 4 }}>
                                        <Typography variant="subtitle1" sx={{ fontWeight: "bold" }}>Overall Summary:</Typography>
                                        <Typography paragraph>{response.course.retention_plan.retention_plan.overall_summary}</Typography>
                                    </Box>
                                )}
                            </Box>
                        )}

                    </Box>
                )}
            </Box>
        </ThemeProvider>
    );
};

export default UploadPage;