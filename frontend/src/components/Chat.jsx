import React, { useState, useRef, useEffect } from 'react';
import axios from 'axios';
import { Send, Upload, FileText, Loader2, Trash2 } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';
import ReactMarkdown from 'react-markdown';

const HOST = import.meta.env.VITE_API_URL || "http://localhost:8003";

const Chat = () => {
    const [messages, setMessages] = useState([
        { role: 'system', content: "👋 Hello! I'm your Personal Knowledge Twin. Upload a document to get started! ✨" }
    ]);
    const [input, setInput] = useState("");
    const [loading, setLoading] = useState(false);
    const [uploading, setUploading] = useState(false);
    const [uploadedFile, setUploadedFile] = useState(null);
    const messagesEndRef = useRef(null);

    const scrollToBottom = () => {
        messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
    };

    useEffect(() => {
        scrollToBottom();
    }, [messages]);

    const handleUpload = async (e) => {
        const file = e.target.files[0];
        if (!file) return;

        setUploading(true);
        const formData = new FormData();
        formData.append("file", file);

        try {
            const res = await axios.post(`${HOST}/upload`, formData);
            setUploadedFile(file.name);

            // 1. Add visual "File Card" to the user's side (Right)
            const fileMsg = {
                role: 'user',
                type: 'file',
                content: file.name,
                isMedia: res.data.is_media,
                mime: res.data.mime,
                preview: res.data.is_media ? URL.createObjectURL(file) : null
            };

            // 2. Add summary from the assistant (Left)
            const summaryMsg = {
                role: 'assistant',
                content: res.data.summary
            };

            setMessages(prev => [...prev, fileMsg, summaryMsg]);
        } catch (err) {
            console.error(err);
            setMessages(prev => [...prev, { role: 'system', content: `❌ Error uploading file: ${err.message}` }]);
        } finally {
            setUploading(false);
        }
    };

    const handleSend = async () => {
        if (!input.trim()) return;

        const userMessage = input;
        setInput("");
        setMessages(prev => [...prev, { role: 'user', content: userMessage }]);
        setLoading(true);

        try {
            const res = await axios.post(`${HOST}/query`, {
                question: userMessage,
                history: messages
            });
            setMessages(prev => [...prev, { role: 'assistant', content: res.data.answer }]);
        } catch (err) {
            console.error(err);
            setMessages(prev => [...prev, { role: 'system', content: `⚠️ Error: ${err.message}` }]);
        } finally {
            setLoading(false);
        }
    };

    const handleClear = async () => {
        try {
            await axios.delete(`${HOST}/clear`);
            setMessages([{ role: 'system', content: "🧹 Knowledge base cleared! Ready for new documents." }]);
            setUploadedFile(null);
        } catch (err) {
            console.error(err);
        }
    };

    return (
        <div className="flex flex-col h-screen max-w-4xl mx-auto p-4 font-sans antialiased text-gray-900">
            <header className="flex justify-between items-center mb-6 py-4 border-b border-gray-100">
                <div className="flex items-center gap-2">
                    <div className="w-8 h-8 bg-gradient-to-tr from-blue-600 to-purple-600 rounded-lg flex items-center justify-center text-white font-bold">KT</div>
                    <h1 className="text-xl font-extrabold tracking-tight text-gray-800">
                        Knowledge Twin
                    </h1>
                </div>
                <div className="flex gap-4">
                    <button
                        onClick={handleClear}
                        className="p-2 text-gray-400 hover:text-red-500 hover:bg-red-50 rounded-lg transition-all"
                        title="Clear Knowledge Base"
                    >
                        <Trash2 size={20} />
                    </button>
                </div>
            </header>

            <div className="flex-1 overflow-y-auto mb-4 space-y-6 pr-2 scrollbar-thin scrollbar-thumb-gray-200">
                <AnimatePresence>
                    {messages.map((msg, idx) => (
                        <motion.div
                            key={idx}
                            initial={{ opacity: 0, y: 10 }}
                            animate={{ opacity: 1, y: 0 }}
                            className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}
                        >
                            <div className={`
                                max-w-[85%] p-5 rounded-2xl shadow-sm leading-relaxed
                                ${msg.role === 'user'
                                    ? 'bg-blue-600 text-white rounded-br-none shadow-md'
                                    : msg.role === 'system'
                                        ? 'bg-gray-100 border border-gray-200 text-gray-700 text-sm'
                                        : 'bg-white border border-gray-200 text-gray-900 rounded-bl-none shadow-sm shadow-gray-100'}
                            `}>
                                <div className={`prose prose-sm max-w-none 
                                    ${msg.role === 'user' ? 'prose-invert font-medium' : 'prose-slate text-gray-800'}
                                    prose-p:leading-relaxed prose-pre:bg-gray-800 prose-pre:text-gray-100
                                    prose-strong:text-inherit
                                `}>
                                    {msg.type === 'file' ? (
                                        <div className="flex flex-col gap-3 py-1">
                                            {msg.isMedia && msg.preview && (
                                                <div className="rounded-xl overflow-hidden shadow-lg bg-black/10">
                                                    {msg.mime.startsWith('image/') ? (
                                                        <img src={msg.preview} alt="Upload" className="max-w-full h-auto object-contain" />
                                                    ) : (
                                                        <video src={msg.preview} controls className="max-w-full h-auto" />
                                                    )}
                                                </div>
                                            )}
                                            <div className="flex items-center gap-3">
                                                <div className="bg-white/20 p-2 rounded-xl">
                                                    <FileText size={24} className="text-white" />
                                                </div>
                                                <div className="flex flex-col">
                                                    <span className="text-white font-bold text-sm tracking-tight">
                                                        {msg.isMedia ? (msg.mime.startsWith('image/') ? 'PHOTO' : 'VIDEO') : 'DOCUMENT'} UPLOADED
                                                    </span>
                                                    <span className="text-blue-100 text-xs opacity-90 truncate max-w-[150px]">{msg.content}</span>
                                                </div>
                                            </div>
                                        </div>
                                    ) : (
                                        <ReactMarkdown>{msg.content}</ReactMarkdown>
                                    )}
                                </div>
                            </div>
                        </motion.div>
                    ))}
                </AnimatePresence>
                {loading && (
                    <motion.div
                        initial={{ opacity: 0 }}
                        animate={{ opacity: 1 }}
                        className="flex justify-start"
                    >
                        <div className="bg-white p-4 rounded-2xl rounded-bl-none shadow-sm border border-gray-100 flex items-center gap-2">
                            <span className="flex gap-1">
                                <span className="w-1.5 h-1.5 bg-blue-400 rounded-full animate-bounce delay-75"></span>
                                <span className="w-1.5 h-1.5 bg-blue-500 rounded-full animate-bounce delay-150"></span>
                                <span className="w-1.5 h-1.5 bg-blue-600 rounded-full animate-bounce delay-300"></span>
                            </span>
                        </div>
                    </motion.div>
                )}
                <div ref={messagesEndRef} />
            </div>

            <div className="bg-white p-5 rounded-3xl shadow-2xl border border-gray-50 mb-2">
                {uploadedFile && (
                    <div className="flex items-center gap-2 mb-4 text-xs text-blue-700 font-semibold bg-blue-50/50 border border-blue-100/50 p-2.5 rounded-xl w-fit">
                        <FileText size={14} />
                        📂 Active Document: {uploadedFile}
                    </div>
                )}

                <div className="flex gap-3 items-end">
                    <label className={`
                        p-3.5 rounded-2xl cursor-pointer transition-all duration-300
                        ${uploading ? 'bg-gray-100 cursor-not-allowed' : 'bg-gray-100 hover:bg-gray-200 text-gray-600 hover:text-blue-600'}
                    `}>
                        <input
                            type="file"
                            accept=".pdf,.txt,image/*,video/*"
                            className="hidden"
                            onChange={handleUpload}
                            disabled={uploading}
                        />
                        {uploading ? <Loader2 className="animate-spin" size={20} /> : <Upload size={20} />}
                    </label>

                    <div className="flex-1 relative">
                        <textarea
                            value={input}
                            onChange={(e) => setInput(e.target.value)}
                            onKeyDown={(e) => {
                                if (e.key === 'Enter' && !e.shiftKey) {
                                    e.preventDefault();
                                    handleSend();
                                }
                            }}
                            placeholder="Ask your twin anything..."
                            className="w-full resize-none border-0 bg-gray-100/80 rounded-2xl px-4 py-3.5 focus:ring-0 focus:bg-gray-200/50 transition-all max-h-32 min-h-[52px] text-gray-800 placeholder-gray-400"
                            rows={1}
                        />
                    </div>

                    <button
                        onClick={handleSend}
                        disabled={!input.trim() || loading}
                        className={`
                            p-3.5 rounded-2xl transition-all duration-300
                            ${!input.trim() || loading
                                ? 'bg-gray-100 text-gray-300 cursor-not-allowed'
                                : 'bg-blue-600 text-white hover:bg-blue-700 shadow-xl shadow-blue-200 active:scale-95'}
                        `}
                    >
                        <Send size={20} />
                    </button>
                </div>
            </div>
        </div>
    );
};

export default Chat;
