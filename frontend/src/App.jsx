import { useState, useRef, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  Rocket, 
  Archive, 
  MessageSquare, 
  Upload, 
  BrainCircuit, 
  Download,
  Save,
  ChevronDown
} from 'lucide-react';
import ReactMarkdown from 'react-markdown';

const API_BASE_URL = import.meta.env.VITE_API_URL || 
  (typeof window !== 'undefined' && (window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1')
    ? 'http://localhost:8000'
    : 'https://smart-hire-1-r5o4.onrender.com');

export default function App() {
  const [activeTab, setActiveTab] = useState('analysis');
  const [isStrictAnonymization, setIsStrictAnonymization] = useState(true);
  const [modelChoice, setModelChoice] = useState('Gemini Flash Lite (Active Pipeline)');

  const [jobDescFile, setJobDescFile] = useState(null);
  const [candidateEmail, setCandidateEmail] = useState('');
  const [resumeFile, setResumeFile] = useState(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [processingStatus, setProcessingStatus] = useState('');
  const [hasResult, setHasResult] = useState(false);
  const [reportData, setReportData] = useState('');

  const [messages, setMessages] = useState([
    { role: 'assistant', content: 'Hello! I am your HR Strategic Assistant. How can I help you regarding the current candidate?' }
  ]);
  const [chatInput, setChatInput] = useState('');

  const [vaultResumes, setVaultResumes] = useState([]);
  const [isLoadingVault, setIsLoadingVault] = useState(false);
  const [searchQuery, setSearchQuery] = useState('');
  const [isSavingToVault, setIsSavingToVault] = useState(false);

  const fileInputRef = useRef(null);
  const jdFileInputRef = useRef(null);

  // Proactively warm up backend if hosted on free-tier Render (prevents cold-start delays)
  useEffect(() => {
    fetch(`${API_BASE_URL}/health`).catch(() => {});
  }, []);

  const loadVault = async () => {
    setIsLoadingVault(true);
    try {
      const response = await fetch(`${API_BASE_URL}/vault`);
      if (!response.ok) return;
      const data = await response.json();
      if (data && data.ids) {
        const list = data.ids.map((id, index) => ({
          id,
          document: data.documents[index],
          metadata: data.metadatas ? data.metadatas[index] : {}
        }));
        setVaultResumes(list);
      }
    } catch (e) {
      console.error("Failed to load vault:", e);
    } finally {
      setIsLoadingVault(false);
    }
  };

  const handleSaveToVault = async () => {
    if (!reportData) return alert("No dossier report available to save!");
    setIsSavingToVault(true);
    try {
      const response = await fetch(`${API_BASE_URL}/vault`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ 
          content: reportData, 
          email: candidateEmail || 'candidate@example.com',
          role: jobDescFile ? jobDescFile.name : 'Selected Role'
        })
      });
      if (!response.ok) {
        let errDetail = `HTTP ${response.status}`;
        try {
          const errJson = await response.json();
          errDetail = errJson.detail || errJson.message || errDetail;
        } catch {}
        alert("Error saving to vault: " + errDetail);
        return;
      }
      const data = await response.json();
      if (data.status === 'success') {
        alert("Successfully saved strategic candidate dossier to the Resume Vault!");
        loadVault();
      } else {
        alert("Error saving to vault: " + (data.message || "Unknown error"));
      }
    } catch (e) {
      console.error("Save to vault error:", e);
      alert(`Failed to connect to API at ${API_BASE_URL}: ${e.message}`);
    } finally {
      setIsSavingToVault(false);
    }
  };

  const handleTabChange = (tab) => {
    setActiveTab(tab);
    if (tab === 'vault') {
      loadVault();
    }
  };

  const handleProcess = async () => {
    if (!jobDescFile || !resumeFile) return alert("Please provide both a Job Description file and a Resume");
    setIsProcessing(true);
    setProcessingStatus("Initializing pipeline & connecting to AI Engine...");
    
    try {
      const formData = new FormData();
      formData.append('job_desc', jobDescFile);
      formData.append('email', candidateEmail || 'candidate@example.com');
      formData.append('resume', resumeFile);
      formData.append('async_mode', 'true');

      let response;
      let attempts = 0;
      const maxAttempts = 2;

      while (attempts < maxAttempts) {
        try {
          response = await fetch(`${API_BASE_URL}/analyze`, {
            method: 'POST',
            body: formData,
          });
          break;
        } catch (fetchErr) {
          attempts++;
          if (attempts < maxAttempts) {
            setProcessingStatus("Backend instance is waking up (Render cold-start)... retrying...");
            await new Promise(r => setTimeout(r, 4000));
          } else {
            throw fetchErr;
          }
        }
      }

      if (!response.ok) {
        let errDetail = `Server error (${response.status})`;
        try {
          const errJson = await response.json();
          if (errJson && (errJson.detail || errJson.message)) {
            errDetail = errJson.detail || errJson.message;
          }
        } catch {}
        alert("Processing Error: " + errDetail);
        return;
      }
      
      const data = await response.json();

      // Case 1: Job was queued asynchronously (prevents gateway/browser timeouts)
      if (data.status === 'queued' && data.job_id) {
        const jobId = data.job_id;
        const statusSteps = [
          'Auditing security guardrails & redacting PII...',
          'Evaluating technical skills against knowledge graph...',
          'Auditing decision fairness & bias metrics...',
          'Assessing market compensation & candidate fit...',
          'Synthesizing final executive hiring dossier...'
        ];
        let stepIdx = 0;
        let completed = false;
        const maxPolls = 100;
        let pollCount = 0;

        while (!completed && pollCount < maxPolls) {
          pollCount++;
          await new Promise(r => setTimeout(r, 2500));
          
          if (stepIdx < statusSteps.length - 1 && pollCount % 4 === 0) {
            stepIdx++;
          }
          setProcessingStatus(statusSteps[stepIdx]);

          try {
            const statusRes = await fetch(`${API_BASE_URL}/analyze/status/${jobId}`);
            if (statusRes.ok) {
              const statusData = await statusRes.json();
              if (statusData.status === 'completed') {
                setReportData(statusData.result);
                setHasResult(true);
                completed = true;
                break;
              } else if (statusData.status === 'failed') {
                alert("Analysis Error: " + (statusData.error || "Processing failed."));
                completed = true;
                break;
              }
            }
          } catch (pollErr) {
            console.warn("Poll attempt transient error:", pollErr);
          }
        }

        if (!completed) {
          alert("The analysis is still processing. Please check back shortly.");
        }
      } 
      // Case 2: Direct synchronous response (fallback)
      else if (data.status === 'success') {
        setReportData(data.result);
        setHasResult(true);
      } else {
        alert("Error: " + (data.message || "Failed to process candidate."));
      }
    } catch (e) {
      console.error("Failed to connect to API:", e);
      alert(`Connection Error (${API_BASE_URL}):\n${e.message || "Unable to reach server"}\n\nNote: If the backend was asleep, it may take 20-30 seconds to wake up. Please click again to retry.`);
    } finally {
      setIsProcessing(false);
      setProcessingStatus('');
    }
  };

  const handleSendMessage = async () => {
    if (!chatInput.trim()) return;
    const newMsg = { role: 'user', content: chatInput };
    setMessages([...messages, newMsg]);
    setChatInput('');
    
    try {
      const response = await fetch(`${API_BASE_URL}/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: chatInput, context: reportData })
      });
      if (!response.ok) {
        throw new Error(`Server returned HTTP ${response.status}`);
      }
      const data = await response.json();
      setMessages(prev => [...prev, { role: 'assistant', content: data.response }]);
    } catch (e) {
      console.error("Chat API error:", e);
      setMessages(prev => [...prev, { role: 'assistant', content: `Connection error with AI (${API_BASE_URL}): ${e.message}` }]);
    }
  };

  const pageVariants = {
    initial: { opacity: 0, y: 20 },
    in: { opacity: 1, y: 0 },
    out: { opacity: 0, y: -20 }
  };

  const pageTransition = {
    type: 'tween',
    ease: 'anticipate',
    duration: 0.5
  };

  return (
    <div className="app-container">
      {/* Sidebar */}
      <div className="sidebar">
        <div style={{ display: 'flex', alignItems: 'center', gap: '1rem', marginBottom: '2rem' }}>
          <BrainCircuit size={40} color="#F5D5E0" />
          <h2 style={{ margin: 0, fontSize: '1.5rem', fontFamily: 'Outfit' }}>Smart Hire</h2>
        </div>

        <nav style={{ flex: 1 }}>
          <div 
            className={`nav-item ${activeTab === 'analysis' ? 'active' : ''}`}
            onClick={() => handleTabChange('analysis')}
          >
            <Rocket size={20} />
            New Analysis
          </div>
          <div 
            className={`nav-item ${activeTab === 'vault' ? 'active' : ''}`}
            onClick={() => handleTabChange('vault')}
          >
            <Archive size={20} />
            Resume Vault
          </div>
          <div 
            className={`nav-item ${activeTab === 'assistant' ? 'active' : ''}`}
            onClick={() => handleTabChange('assistant')}
          >
            <MessageSquare size={20} />
            HR Assistant
          </div>
        </nav>

        <div style={{ marginTop: 'auto' }}>
          <hr style={{ borderColor: 'rgba(255,255,255,0.1)', margin: '1rem 0' }} />
          
          <div style={{ marginBottom: '1rem' }}>
            <label style={{ fontSize: '0.9rem', marginBottom: '0.5rem', display: 'block', opacity: 0.8 }}>Intelligence Engine</label>
            <div style={{ position: 'relative' }}>
              <select 
                className="glass-input" 
                value={modelChoice} 
                onChange={(e) => setModelChoice(e.target.value)}
                style={{ appearance: 'none', cursor: 'pointer', paddingRight: '2.5rem' }}
              >
                <option value="Gemini Flash Lite (Active Pipeline)">Gemini Flash Lite</option>
                <option value="Gemini 2.0 Flash (Restricted Quota)">Gemini 2.0 Flash</option>
                <option value="Gemini 1.5 Pro (Authwall)">Gemini 1.5 Pro</option>
              </select>
              <ChevronDown size={16} style={{ position: 'absolute', right: '1rem', top: '50%', transform: 'translateY(-50%)', pointerEvents: 'none', opacity: 0.7 }} />
            </div>
          </div>

          <div className="toggle-container">
            <span style={{ fontSize: '0.95rem' }}>Strict Anonymization</span>
            <label className="toggle-switch">
              <input 
                type="checkbox" 
                checked={isStrictAnonymization} 
                onChange={() => setIsStrictAnonymization(!isStrictAnonymization)}
              />
              <span className="slider"></span>
            </label>
          </div>
        </div>
      </div>

      {/* Main Content */}
      <div className="main-content">
        <header style={{ marginBottom: '3rem' }}>
          <h1 className="main-header">Smart Hire Intelligence</h1>
          <p className="subtitle">AI-Powered Recruitment Infrastructure & Bias-Free Assessment</p>
          
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '1.5rem', marginTop: '2rem' }}>
            <div className="metric-card">
              <div className="metric-value">1</div>
              <div className="metric-label">Candidates Processed</div>
            </div>
            <div className="metric-card">
              <div className="metric-value">{hasResult ? 'Calculated' : '--'}</div>
              <div className="metric-label">Match Score Status</div>
            </div>
            <div className="metric-card">
              <div className="metric-value">{hasResult ? 'Analyzed' : 'Pending'}</div>
              <div className="metric-label">Hiring Difficulty</div>
            </div>
          </div>
        </header>

        <AnimatePresence mode="wait">
          <motion.div
            key={activeTab}
            initial="initial"
            animate="in"
            exit="out"
            variants={pageVariants}
            transition={pageTransition}
          >
            {activeTab === 'analysis' && (
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 2fr', gap: '2rem' }}>
                <div className="glass-card">
                  <h3 style={{ marginTop: 0, marginBottom: '1.5rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                    <Rocket size={20} /> Opportunity Profile
                  </h3>
                  <label style={{ display: 'block', marginBottom: '0.5rem', opacity: 0.9 }}>Upload Job Description</label>
                  <div 
                      onClick={() => jdFileInputRef.current.click()}
                      style={{ 
                        border: '2px dashed rgba(255,255,255,0.2)', 
                        padding: '2rem', 
                        borderRadius: '12px', 
                        textAlign: 'center',
                        background: jobDescFile ? 'rgba(102, 103, 171, 0.2)' : 'rgba(0,0,0,0.1)',
                        cursor: 'pointer',
                        transition: 'all 0.3s'
                      }}
                    >
                      <input 
                        type="file" 
                        accept=".pdf,.txt,.md" 
                        onChange={(e) => setJobDescFile(e.target.files[0])} 
                        style={{ display: 'none' }} 
                        ref={jdFileInputRef} 
                      />
                      <Upload size={32} color="#6667AB" style={{ marginBottom: '1rem' }} />
                      <div style={{ fontWeight: 600 }}>{jobDescFile ? jobDescFile.name : 'Upload Job Description'}</div>
                      <div style={{ fontSize: '0.8rem', opacity: 0.7, marginTop: '0.5rem' }}>Drag & drop or click to browse</div>
                    </div>
                </div>

                <div className="glass-card">
                  <h3 style={{ marginTop: 0, marginBottom: '1.5rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                    <Upload size={20} /> Candidate Inputs
                  </h3>
                  
                  <div style={{ marginBottom: '1.5rem' }}>
                    <label style={{ display: 'block', marginBottom: '0.5rem', opacity: 0.9 }}>Contact Address</label>
                    <input 
                      type="email" 
                      className="glass-input" 
                      placeholder="candidate@example.com"
                      value={candidateEmail}
                      onChange={(e) => setCandidateEmail(e.target.value)}
                    />
                  </div>

                  <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1.5rem', marginBottom: '2rem' }}>
                    <div 
                      onClick={() => fileInputRef.current.click()}
                      style={{ 
                        border: '2px dashed rgba(255,255,255,0.2)', 
                        padding: '2rem', 
                        borderRadius: '12px', 
                        textAlign: 'center',
                        background: resumeFile ? 'rgba(102, 103, 171, 0.2)' : 'rgba(0,0,0,0.1)',
                        cursor: 'pointer',
                        transition: 'all 0.3s'
                      }}
                    >
                      <input 
                        type="file" 
                        accept=".pdf" 
                        onChange={(e) => setResumeFile(e.target.files[0])} 
                        style={{ display: 'none' }} 
                        ref={fileInputRef} 
                      />
                      <Upload size={32} color="#6667AB" style={{ marginBottom: '1rem' }} />
                      <div style={{ fontWeight: 600 }}>{resumeFile ? resumeFile.name : 'Upload Resume (PDF)'}</div>
                      <div style={{ fontSize: '0.8rem', opacity: 0.7, marginTop: '0.5rem' }}>Drag & drop or click to browse</div>
                    </div>
                    <div style={{ 
                      border: '2px dashed rgba(255,255,255,0.2)', 
                      padding: '2rem', 
                      borderRadius: '12px', 
                      textAlign: 'center',
                      background: 'rgba(0,0,0,0.1)'
                    }}>
                      <Upload size={32} color="#6667AB" style={{ marginBottom: '1rem' }} />
                      <div style={{ fontWeight: 600 }}>Intro Video (Optional)</div>
                      <div style={{ fontSize: '0.8rem', opacity: 0.7, marginTop: '0.5rem' }}>Integrity checking supported</div>
                    </div>
                  </div>

                  <button 
                    className="glass-button" 
                    style={{ width: '100%', padding: '1rem', fontSize: '1.05rem', display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '0.75rem' }}
                    onClick={handleProcess}
                    disabled={isProcessing}
                  >
                    {isProcessing ? (
                      <>
                        <motion.div
                          animate={{ rotate: 360 }}
                          transition={{ repeat: Infinity, duration: 1, ease: 'linear' }}
                          style={{ display: 'inline-flex' }}
                        >
                          <BrainCircuit size={22} />
                        </motion.div>
                        <span style={{ fontSize: '0.95rem' }}>{processingStatus || 'Analyzing Candidate...'}</span>
                      </>
                    ) : (
                      <>
                        <BrainCircuit size={22} /> Run Intelligence Engine
                      </>
                    )}
                  </button>

                  {hasResult && !isProcessing && (
                    <motion.div 
                      initial={{ opacity: 0, y: 10 }}
                      animate={{ opacity: 1, y: 0 }}
                      style={{ marginTop: '2rem', padding: '1.5rem', background: 'rgba(0,0,0,0.2)', borderRadius: '12px', maxHeight: '400px', overflowY: 'auto' }}
                    >
                      <h4 style={{ margin: '0 0 1rem 0', color: '#ffb3c6' }}>Strategic Dossier Generated</h4>
                      <div style={{ fontSize: '0.9rem', lineHeight: 1.5, opacity: 0.9 }}>
                        <ReactMarkdown>{reportData}</ReactMarkdown>
                      </div>
                      <div style={{ display: 'flex', gap: '1rem', marginTop: '1.5rem' }}>
                        <button className="glass-button" onClick={() => {
                          const blob = new Blob([reportData], { type: 'text/markdown' });
                          const url = URL.createObjectURL(blob);
                          const a = document.createElement('a');
                          a.href = url;
                          a.download = `Strategic_Dossier_${candidateEmail || 'Candidate'}.md`;
                          a.click();
                        }}>
                          <Download size={18} /> Download Dossier
                        </button>
                        <button 
                          className="glass-button outline"
                          onClick={handleSaveToVault}
                          disabled={isSavingToVault}
                        >
                          <Save size={18} /> {isSavingToVault ? 'Saving...' : 'Save to Vault'}
                        </button>
                      </div>
                    </motion.div>
                  )}
                </div>
              </div>
            )}

            {activeTab === 'vault' && (
              <div className="glass-card" style={{ minHeight: '500px' }}>
                <h3 style={{ marginTop: 0, marginBottom: '1.5rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                  <Archive size={20} /> Talent Archive
                </h3>
                
                <div style={{ display: 'flex', gap: '1rem', marginBottom: '2rem' }}>
                  <input 
                    type="text" 
                    className="glass-input" 
                    placeholder="🔍 Search vault by skills, email, or ID..."
                    value={searchQuery}
                    onChange={(e) => setSearchQuery(e.target.value)}
                  />
                  <button className="glass-button" onClick={loadVault} disabled={isLoadingVault} style={{ whiteSpace: 'nowrap' }}>
                    Refresh List
                  </button>
                </div>

                {isLoadingVault ? (
                  <div style={{ textAlign: 'center', marginTop: '4rem', opacity: 0.7 }}>
                    <motion.div
                      animate={{ rotate: 360 }}
                      transition={{ repeat: Infinity, duration: 1, ease: 'linear' }}
                      style={{ display: 'inline-block', marginBottom: '1rem' }}
                    >
                      <BrainCircuit size={32} />
                    </motion.div>
                    <p>Loading database archive...</p>
                  </div>
                ) : (
                  <div className="vault-list" style={{ display: 'flex', flexDirection: 'column', gap: '1rem', maxHeight: '500px', overflowY: 'auto' }}>
                    {vaultResumes.filter(item => {
                      const q = searchQuery.toLowerCase();
                      return (
                        item.id.toLowerCase().includes(q) ||
                        (item.document && item.document.toLowerCase().includes(q)) ||
                        (item.metadata.email && item.metadata.email.toLowerCase().includes(q)) ||
                        (item.metadata.role && item.metadata.role.toLowerCase().includes(q))
                      );
                    }).length === 0 ? (
                      <div style={{ textAlign: 'center', opacity: 0.5, marginTop: '4rem' }}>
                        <Archive size={48} style={{ marginBottom: '1rem' }} />
                        <p>No candidates found in the database vault.</p>
                      </div>
                    ) : (
                      vaultResumes.filter(item => {
                        const q = searchQuery.toLowerCase();
                        return (
                          item.id.toLowerCase().includes(q) ||
                          (item.document && item.document.toLowerCase().includes(q)) ||
                          (item.metadata.email && item.metadata.email.toLowerCase().includes(q)) ||
                          (item.metadata.role && item.metadata.role.toLowerCase().includes(q))
                        );
                      }).map((item) => (
                        <div key={item.id} className="glass-card" style={{ padding: '1.5rem', background: 'rgba(255,255,255,0.03)', border: '1px solid rgba(255,255,255,0.05)', textAlign: 'left' }}>
                          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1rem' }}>
                            <h4 style={{ margin: 0, color: '#ffb3c6' }}>ID: {item.id}</h4>
                            <span style={{ fontSize: '0.8rem', opacity: 0.7 }}>
                              {item.metadata.timestamp ? new Date(item.metadata.timestamp * 1000).toLocaleDateString() : 'Archive'}
                            </span>
                          </div>
                          <p style={{ margin: '0 0 0.5rem 0', fontSize: '0.95rem' }}><strong>Email:</strong> {item.metadata.email || 'Unknown'}</p>
                          {item.metadata.role && (
                            <p style={{ margin: '0 0 1rem 0', fontSize: '0.95rem' }}><strong>Source/Role:</strong> {item.metadata.role}</p>
                          )}
                          <details style={{ cursor: 'pointer' }}>
                            <summary style={{ fontSize: '0.9rem', color: '#ffb3c6', outline: 'none' }}>View Resume Details</summary>
                            <pre style={{ whiteSpace: 'pre-wrap', fontSize: '0.8rem', background: 'rgba(0,0,0,0.2)', padding: '1rem', borderRadius: '8px', marginTop: '0.5rem', maxHeight: '150px', overflowY: 'auto', color: '#f5d5e0', fontFamily: 'monospace' }}>
                              {item.document}
                            </pre>
                          </details>
                        </div>
                      ))
                    )}
                  </div>
                )}
              </div>
            )}

            {activeTab === 'assistant' && (
              <div className="glass-card" style={{ height: '600px', display: 'flex', flexDirection: 'column' }}>
                <h3 style={{ marginTop: 0, marginBottom: '1.5rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                  <MessageSquare size={20} /> HR Strategic Assistant
                </h3>
                
                <div style={{ flex: 1, overflowY: 'auto', padding: '1rem', background: 'rgba(0,0,0,0.1)', borderRadius: '12px', marginBottom: '1.5rem' }}>
                  {messages.map((msg, idx) => (
                    <motion.div 
                      initial={{ opacity: 0, x: msg.role === 'user' ? 20 : -20 }}
                      animate={{ opacity: 1, x: 0 }}
                      key={idx} 
                      className={`chat-message ${msg.role}`}
                    >
                      <div className="avatar">
                        {msg.role === 'user' ? 'U' : <BrainCircuit size={20} />}
                      </div>
                      <div className="message-content">
                        <ReactMarkdown>{msg.content}</ReactMarkdown>
                      </div>
                    </motion.div>
                  ))}
                </div>

                <div style={{ display: 'flex', gap: '1rem' }}>
                  <input 
                    type="text" 
                    className="glass-input" 
                    placeholder="Ask about candidate fit, suggest interview questions..."
                    value={chatInput}
                    onChange={(e) => setChatInput(e.target.value)}
                    onKeyDown={(e) => e.key === 'Enter' && handleSendMessage()}
                  />
                  <button className="glass-button" onClick={handleSendMessage}>Send</button>
                </div>
              </div>
            )}
          </motion.div>
        </AnimatePresence>
      </div>
    </div>
  );
}
