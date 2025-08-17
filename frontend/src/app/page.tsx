"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';

type Message = {
	sender: string;
	message: string;
	timestamp?: string;
	context?: Record<string, unknown> | null;
};

type SuggestionStatus = 'proposed' | 'accepted' | 'dismissed' | 'in_progress' | 'completed';
type Suggestion = {
	id: string;
	user_id?: string;
	agent: string;
	title: string;
	details: string;
	category: string;
	status: SuggestionStatus;
	created_at?: string;
	conversation_id?: string;
	message_index?: number;
	message_timestamp?: string;
};

type Issue = {
	id: string;
	title: string;
	details: string;
	category: string;
	severity?: string;
	message_index?: number;
	message_timestamp?: string;
	created_at?: string;
	status?: string;
    priority?: string;
    time_window?: string;
};

type Episode = {
	id: string;
	title: string;
	trigger_type: string;
	trigger_description: string;
	status: string;
	priority: number;
	member_state_before?: string;
	member_state_after?: string;
	created_at?: string;
};

type Decision = {
	id: string;
	type: string;
	content: string;
	timestamp: string;
	responsible_agent: string;
	rationale?: string;
};

type Experiment = {
	id: string;
	hypothesis: string;
	status: string;
	template?: string;
	member_id: string;
	outcome?: string;
	created_at?: string;
};

export default function ElyxDashboard() {
	// Navigation state
	const [activeTab, setActiveTab] = useState("chat");

	// Data states
	const [messages, setMessages] = useState<Message[]>([]);
	const [suggestions, setSuggestions] = useState<Suggestion[]>([]);
	const [issues, setIssues] = useState<Issue[]>([]);
	const [episodes, setEpisodes] = useState<Episode[]>([]);
	const [decisions, setDecisions] = useState<Decision[]>([]);
	const [experiments, setExperiments] = useState<Experiment[]>([]);
	const [healthMetrics, setHealthMetrics] = useState<any[]>([]);
	const [agentPerformance, setAgentPerformance] = useState<any>({});
	const [traces, setTraces] = useState<any[]>([]);

	// UI states
	const [inputMessage, setInputMessage] = useState("");
	const [isLoading, setIsLoading] = useState(false);

	const [planFilter, setPlanFilter] = useState<'todo' | 'in_progress' | 'completed'>('todo');

	// Form states for data management
	const [showEpisodeForm, setShowEpisodeForm] = useState(false);
	const [showExperimentForm, setShowExperimentForm] = useState(false);
	const [episodeForm, setEpisodeForm] = useState({
		title: '',
		trigger_type: 'user_message',
		trigger_description: '',
		priority: 2,
		member_state_before: ''
	});
	const [experimentForm, setExperimentForm] = useState({
		hypothesis: '',
		template: '',
		member_id: 'rohan'
	});

	const messagesEndRef = useRef<HTMLDivElement>(null);
	const backendBase = "http://localhost:8787";

	// Scroll to bottom of messages
	const scrollToBottom = () => {
		messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
	};

	useEffect(() => {
		scrollToBottom();
	}, [messages]);

	// Load all data
	const refreshAll = async () => {
		try {
			const [messagesRes, suggestionsRes, issuesRes, episodesRes, decisionsRes, experimentsRes, healthMetricsRes, agentPerformanceRes, tracesRes] = await Promise.all([
				fetch(`${backendBase}/messages`),
				fetch(`${backendBase}/suggestions`),
				fetch(`${backendBase}/issues`),
				fetch(`${backendBase}/episodes`),
				fetch(`${backendBase}/decisions`),
				fetch(`${backendBase}/experiments`),
				fetch(`${backendBase}/health-metrics`),
				fetch(`${backendBase}/agents/performance`),
				fetch(`${backendBase}/traces`)
			]);

			if (messagesRes.ok) {
				const data = await messagesRes.json();
				setMessages(data as Message[]);
			}
			if (suggestionsRes.ok) {
				const data = await suggestionsRes.json();
				setSuggestions(data as Suggestion[]);
			}
			if (issuesRes.ok) {
				const data = await issuesRes.json();
				setIssues(data as Issue[]);
			}
			if (episodesRes.ok) {
				const data = await episodesRes.json();
				setEpisodes(data as Episode[]);
			}
			if (decisionsRes.ok) {
				const data = await decisionsRes.json();
				setDecisions(data as Decision[]);
			}
			if (experimentsRes.ok) {
				const data = await experimentsRes.json();
				setExperiments(data as Experiment[]);
			}
			if (healthMetricsRes.ok) {
				const data = await healthMetricsRes.json();
				setHealthMetrics(data);
			}
			if (agentPerformanceRes.ok) {
				const data = await agentPerformanceRes.json();
				setAgentPerformance(data);
			}
			if (tracesRes.ok) {
				const data = await tracesRes.json();
				setTraces(data);
			}
		} catch (error) {
			console.error('Error loading data:', error);
		}
	};

	useEffect(() => {
		refreshAll();
	}, []);

	// Send message
	const sendMessage = async (e: React.FormEvent) => {
		e.preventDefault();
		if (!inputMessage.trim() || isLoading) return;

		setIsLoading(true);
		const userMessage = inputMessage.trim();
		setInputMessage("");

		try {
			const response = await fetch(`${backendBase}/chat`, {
				method: "POST",
				headers: { "Content-Type": "application/json" },
				body: JSON.stringify({
					sender: "Rohan",
					message: userMessage,
					use_crewai: false,
				}),
			});

			if (response.ok) {
				const newMessages = await response.json();
				setMessages(newMessages as Message[]);
				// Refresh suggestions and issues after chat
				setTimeout(refreshAll, 500);
			}
		} catch (error) {
			console.error("Error sending message:", error);
		} finally {
			setIsLoading(false);
		}
	};

	// Add episode
	const addEpisode = async (e: React.FormEvent) => {
		e.preventDefault();
		try {
			const response = await fetch(`${backendBase}/episodes`, {
				method: "POST",
				headers: { "Content-Type": "application/json" },
				body: JSON.stringify(episodeForm),
			});

			if (response.ok) {
				setShowEpisodeForm(false);
				setEpisodeForm({
					title: '',
					trigger_type: 'user_message',
					trigger_description: '',
					priority: 2,
					member_state_before: ''
				});
				refreshAll();
			}
		} catch (error) {
			console.error("Error adding episode:", error);
		}
	};

	// Add experiment
	const addExperiment = async (e: React.FormEvent) => {
		e.preventDefault();
		try {
			const response = await fetch(`${backendBase}/experiments`, {
				method: "POST",
				headers: { "Content-Type": "application/json" },
				body: JSON.stringify(experimentForm),
			});

			if (response.ok) {
				setShowExperimentForm(false);
				setExperimentForm({
					hypothesis: '',
					template: '',
					member_id: 'rohan'
				});
				refreshAll();
			}
		} catch (error) {
			console.error("Error adding experiment:", error);
		}
	};

	// Generate mock data
	const generateMockData = async () => {
		try {
			const response = await fetch(`${backendBase}/demo/generate-mock`, {
				method: "POST",
			});
			if (response.ok) {
				refreshAll();
			}
		} catch (error) {
			console.error("Error generating mock data:", error);
		}
	};

	// Update suggestion status
	const updateSuggestionStatus = async (suggestionId: string, status: string) => {
		try {
			const response = await fetch(`${backendBase}/suggestions/${suggestionId}/status`, {
				method: "POST",
				headers: { "Content-Type": "application/json" },
				body: JSON.stringify({ status }),
			});
			
			if (response.ok) {
				// Refresh suggestions to get updated status
				const suggestionsRes = await fetch(`${backendBase}/suggestions`);
				if (suggestionsRes.ok) {
					const data = await suggestionsRes.json();
					setSuggestions(data as Suggestion[]);
				}
			}
		} catch (error) {
			console.error("Error updating suggestion status:", error);
		}
	};



	// Plan items (accepted suggestions organized by status)
	const planItems = useMemo(() => {
		const accepted = suggestions.filter(s => s.status === 'accepted' || s.status === 'in_progress' || s.status === 'completed');
		return {
			todo: accepted.filter(s => s.status === 'accepted'),
			in_progress: accepted.filter(s => s.status === 'in_progress'), 
			completed: accepted.filter(s => s.status === 'completed')
		};
	}, [suggestions]);

	// New/pending suggestions (not yet accepted/dismissed)
	const pendingSuggestions = useMemo(() => {
		return suggestions.filter(s => s.status === 'proposed');
	}, [suggestions]);

	// Get status badge class
	const getStatusBadge = (status: string) => {
		switch (status) {
			case 'completed': return 'badge badge-success';
			case 'in_progress': return 'badge badge-warning';
			case 'accepted': return 'badge badge-primary';
			case 'running': return 'badge badge-warning';
			case 'planned': return 'badge badge-secondary';
			default: return 'badge badge-primary';
		}
	};

	// Get priority badge class
	const getPriorityBadge = (priority: number) => {
		if (priority >= 4) return 'badge badge-danger';
		if (priority >= 3) return 'badge badge-warning';
		return 'badge badge-success';
	};

		// Stats
	const stats = {
		totalMessages: messages.length,
		pendingSuggestions: pendingSuggestions.length,
		activeTasks: planItems.in_progress.length,
		completedTasks: planItems.completed.length,
		openIssues: issues.filter(i => i.status !== 'resolved').length,
		activeExperiments: experiments.filter(e => e.status === 'running').length,
    };

	return (
		<div className="app-container">
			{/* Header */}
			<header className="app-header">
				<div className="header-content">
					<div className="logo">Elyx Health Platform</div>
					<div className="flex items-center gap-4">
						<span className="text-sm text-gray-500">Member: Rohan Patel</span>
					</div>
				</div>
			</header>

			{/* Main Content */}
			<main className="main-content">
				{/* Navigation */}
				<nav className="nav-tabs">
					<button
						className={`nav-tab ${activeTab === "chat" ? "active" : ""}`}
						onClick={() => setActiveTab("chat")}
					>
						💬 Chat
					</button>
					<button
						className={`nav-tab ${activeTab === "dashboard" ? "active" : ""}`}
						onClick={() => setActiveTab("dashboard")}
					>
						📋 My Plan
					</button>
					<button
						className={`nav-tab ${activeTab === "data" ? "active" : ""}`}
						onClick={() => setActiveTab("data")}
					>
						🗄️ Data Management
					</button>
					<button
						className={`nav-tab ${activeTab === "analytics" ? "active" : ""}`}
						onClick={() => setActiveTab("analytics")}
					>
						📈 Analytics
					</button>
				</nav>

				{/* Chat Tab */}
				{activeTab === "chat" && (
					<div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
						{/* Chat Interface */}
						<div className="lg:col-span-2">
							<div className="card">
								<div className="card-header">
									<h2 className="card-title">AI Health Team Chat</h2>
									<p className="card-subtitle">Chat with Ruby, Dr. Warren, Advik, Carla, Rachel, and Neel</p>
								</div>
								<div className="chat-container">
									<div className="chat-messages">
										{messages.map((msg, idx) => (
											<div key={idx} className={`message ${msg.sender === 'Rohan' ? 'user' : 'agent'}`}>
												<div className="message-header">
													{msg.sender} {msg.timestamp && `• ${new Date(msg.timestamp).toLocaleTimeString()}`}
												</div>
												<div className="message-content">
													{msg.message}
												</div>
							</div>
										))}
										{isLoading && (
											<div className="message agent">
												<div className="message-header">AI Team</div>
												<div className="message-content">
													<div className="loading"></div> Thinking...
												</div>
											</div>
										)}
										<div ref={messagesEndRef} />
									</div>
									<div className="chat-input-container">
										<form onSubmit={sendMessage} className="chat-input-form">
											<input
												type="text"
												value={inputMessage}
												onChange={(e) => setInputMessage(e.target.value)}
												placeholder="Ask about your health, nutrition, fitness, or schedule..."
												className="form-input chat-input"
												disabled={isLoading}
											/>
											<button type="submit" className="btn btn-primary" disabled={isLoading}>
												{isLoading ? <div className="loading"></div> : "Send"}
											</button>
										</form>
									</div>
								</div>
							</div>
						</div>

						{/* Quick Stats */}
						<div className="space-y-6">
							<div className="card">
								<div className="card-header">
									<h3 className="card-title">Quick Stats</h3>
								</div>
								<div className="space-y-4">
									<div className="flex justify-between items-center">
										<span className="text-sm text-gray-600">Messages</span>
										<span className="font-semibold">{stats.totalMessages}</span>
									</div>
									<div className="flex justify-between items-center">
										<span className="text-sm text-gray-600">New Suggestions</span>
										<span className="font-semibold text-orange-600">{stats.pendingSuggestions}</span>
									</div>
									<div className="flex justify-between items-center">
										<span className="text-sm text-gray-600">Active Tasks</span>
										<span className="font-semibold text-blue-600">{stats.activeTasks}</span>
									</div>
									<div className="flex justify-between items-center">
										<span className="text-sm text-gray-600">Completed</span>
										<span className="font-semibold text-green-600">{stats.completedTasks}</span>
									</div>
								</div>
							</div>

							<div className="card">
								<div className="card-header">
									<h3 className="card-title">New Suggestions</h3>
									<p className="card-subtitle">Review and accept/reject AI recommendations</p>
								</div>
								<div className="space-y-3 max-h-96 overflow-y-auto">
									{pendingSuggestions.map((suggestion) => (
										<div key={suggestion.id} className="p-4 border border-gray-200 rounded-lg">
											<div className="flex items-start justify-between mb-2">
												<h4 className="text-sm font-medium">{suggestion.title}</h4>
												<span className="text-xs text-gray-500">{suggestion.agent}</span>
											</div>
											<p className="text-xs text-gray-600 mb-3">{suggestion.details}</p>
											<div className="flex items-center justify-between">
												<span className="text-xs text-gray-500">{suggestion.category}</span>
							<div className="flex gap-2">
													<button
														onClick={() => updateSuggestionStatus(suggestion.id, 'accept')}
														className="btn btn-sm btn-secondary"
													>
														✓ Accept
													</button>
													<button
														onClick={() => updateSuggestionStatus(suggestion.id, 'dismiss')}
														className="btn btn-sm"
													>
														✗ Dismiss
													</button>
												</div>
											</div>
										</div>
									))}
									{pendingSuggestions.length === 0 && (
										<p className="text-sm text-gray-500 text-center py-8">No new suggestions. Start chatting to get personalized recommendations!</p>
									)}
								</div>
							</div>
						</div>
					</div>
				)}

				{/* Dashboard Tab */}
				{activeTab === "dashboard" && (
						<div>
						{/* Stats Overview */}
						<div className="stats-grid">
							<div className="stat-card">
								<div className="stat-value text-orange-500">{stats.pendingSuggestions}</div>
								<div className="stat-label">New Suggestions</div>
							</div>
							<div className="stat-card">
								<div className="stat-value text-blue-500">{planItems.todo.length}</div>
								<div className="stat-label">To-Do Tasks</div>
							</div>
							<div className="stat-card">
								<div className="stat-value text-purple-500">{stats.activeTasks}</div>
								<div className="stat-label">In Progress</div>
							</div>
							<div className="stat-card">
								<div className="stat-value text-green-500">{stats.completedTasks}</div>
								<div className="stat-label">Completed</div>
							</div>
						</div>

						<div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
							{/* Plan */}
							<div className="card">
								<div className="card-header">
									<h3 className="card-title">Your Health Plan</h3>
									<div className="flex gap-2">
										{(['todo', 'in_progress', 'completed'] as const).map((status) => (
											<button
												key={status}
												className={`btn btn-sm ${planFilter === status ? 'btn-primary' : ''}`}
												onClick={() => setPlanFilter(status)}
											>
												{status === 'todo' ? `To-Do (${planItems.todo.length})` : 
												 status === 'in_progress' ? `In Progress (${planItems.in_progress.length})` :
												 `Completed (${planItems.completed.length})`}
											</button>
										))}
									</div>
								</div>
								<div className="space-y-3 max-h-96 overflow-y-auto">
									{planItems[planFilter].map((item) => (
										<div key={item.id} className="p-4 border border-gray-200 rounded-lg">
											<div className="flex items-start justify-between mb-2">
												<h4 className="font-medium">{item.title}</h4>
												<div className="flex items-center gap-2">
													<span className={getStatusBadge(item.status)}>
														{item.status}
													</span>
													{planFilter === 'todo' && (
														<button
															onClick={() => updateSuggestionStatus(item.id, 'in_progress')}
															className="btn btn-sm btn-primary"
														>
															Start
														</button>
													)}
													{planFilter === 'in_progress' && (
														<button
															onClick={() => updateSuggestionStatus(item.id, 'completed')}
															className="btn btn-sm btn-secondary"
														>
															Complete
														</button>
													)}
												</div>
											</div>
											<p className="text-sm text-gray-600 mb-3">{item.details}</p>
											<div className="flex items-center justify-between text-xs text-gray-500">
												<span>by {item.agent}</span>
												<span>{item.category}</span>
											</div>
										</div>
									))}
									{planItems[planFilter].length === 0 && (
										<div className="text-center py-8">
											<p className="text-sm text-gray-500 mb-2">
												{planFilter === 'todo' ? 'No tasks to do' :
												 planFilter === 'in_progress' ? 'No tasks in progress' :
												 'No completed tasks'}
											</p>
											{planFilter === 'todo' && (
												<p className="text-xs text-gray-400">Accept suggestions from the chat to add them to your plan</p>
											)}
										</div>
									)}
								</div>
							</div>

							{/* Issues */}
							<div className="card">
								<div className="card-header">
									<h3 className="card-title">Issues & Concerns</h3>
								</div>
								<div className="space-y-3 max-h-96 overflow-y-auto">
									{issues.filter(i => (i.status || 'open') !== 'resolved').map((issue) => (
										<div key={issue.id} className="p-4 border border-gray-200 rounded-lg">
											<div className="flex items-start justify-between mb-2">
												<h4 className="font-medium">{issue.title}</h4>
												<div className="flex items-center gap-2">
													{issue.priority && (
														<span className="badge badge-primary">{issue.priority}</span>
													)}
													{issue.time_window && (
														<span className="badge badge-secondary">{issue.time_window}</span>
													)}
													<span className={`badge ${issue.severity === 'high' ? 'badge-danger' : issue.severity === 'medium' ? 'badge-warning' : 'badge-success'}`}>
														{issue.severity || 'low'}
													</span>
												</div>
											</div>
											<p className="text-sm text-gray-600 mb-3">{issue.details}</p>
											<div className="text-xs text-gray-500">{issue.category}</div>
										</div>
									))}
									{issues.length === 0 && (
										<p className="text-sm text-gray-500 text-center py-8">No issues reported.</p>
									)}
								</div>
							</div>

							{/* Episodes */}
							<div className="card">
								<div className="card-header">
									<h3 className="card-title">Episodes</h3>
								</div>
								<div className="space-y-3 max-h-96 overflow-y-auto">
									{episodes.map((episode) => (
										<div key={episode.id} className="p-4 border border-gray-200 rounded-lg">
											<div className="flex items-start justify-between mb-2">
												<h4 className="font-medium">{episode.title}</h4>
												<div className="flex items-center gap-2">
													<span className={getPriorityBadge(episode.priority)}>
														P{episode.priority}
													</span>
													<span className={getStatusBadge(episode.status)}>
														{episode.status}
													</span>
												</div>
											</div>
											<p className="text-sm text-gray-600 mb-2">{episode.trigger_description}</p>
											<div className="text-xs text-gray-500">{episode.trigger_type}</div>
										</div>
									))}
									{episodes.length === 0 && (
										<p className="text-sm text-gray-500 text-center py-8">No episodes recorded.</p>
									)}
								</div>
							</div>

							{/* Experiments */}
							<div className="card">
								<div className="card-header">
									<h3 className="card-title">Experiments</h3>
								</div>
								<div className="space-y-3 max-h-96 overflow-y-auto">
									{experiments.map((experiment) => (
										<div key={experiment.id} className="p-4 border border-gray-200 rounded-lg">
											<div className="flex items-start justify-between mb-2">
												<h4 className="font-medium text-sm">{experiment.hypothesis}</h4>
												<span className={getStatusBadge(experiment.status)}>
													{experiment.status}
												</span>
											</div>
											{experiment.template && (
												<div className="text-xs text-gray-500 mb-2">Template: {experiment.template}</div>
											)}
											{experiment.outcome && (
												<p className="text-sm text-gray-600">{experiment.outcome}</p>
											)}
										</div>
									))}
									{experiments.length === 0 && (
										<p className="text-sm text-gray-500 text-center py-8">No experiments running.</p>
									)}
								</div>
							</div>
						</div>
						</div>
				)}

				{/* Data Management Tab */}
				{activeTab === "data" && (
					<div>
						<div className="flex items-center justify-between mb-6">
							<h2 className="text-2xl font-bold">Data Management</h2>
							<div className="flex gap-2">
								<button onClick={generateMockData} className="btn btn-secondary">
									🎲 Generate Mock Data
								</button>
								<button
									onClick={async () => {
										await fetch(`${backendBase}/reset-soft`, { method: 'POST' });
										await refreshAll();
									}}
									className="btn"
								>
									🧹 Reset (keep user)
								</button>
							</div>
						</div>

						<div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
							{/* Add Episode */}
							<div className="card">
								<div className="card-header">
									<h3 className="card-title">Add New Episode</h3>
									<button
										onClick={() => setShowEpisodeForm(!showEpisodeForm)}
										className="btn btn-primary btn-sm"
									>
										{showEpisodeForm ? 'Cancel' : '+ Add Episode'}
									</button>
								</div>
								{showEpisodeForm && (
									<form onSubmit={addEpisode} className="space-y-4">
										<div className="form-group">
											<label className="form-label">Title</label>
											<input
												type="text"
												className="form-input"
												value={episodeForm.title}
												onChange={(e) => setEpisodeForm({...episodeForm, title: e.target.value})}
												required
											/>
										</div>
										<div className="form-group">
											<label className="form-label">Trigger Type</label>
											<select
												className="form-input form-select"
												value={episodeForm.trigger_type}
												onChange={(e) => setEpisodeForm({...episodeForm, trigger_type: e.target.value})}
											>
												<option value="user_message">User Message</option>
												<option value="health_alert">Health Alert</option>
												<option value="scheduled_check">Scheduled Check</option>
												<option value="external_data">External Data</option>
											</select>
										</div>
										<div className="form-group">
											<label className="form-label">Description</label>
											<textarea
												className="form-input form-textarea"
												value={episodeForm.trigger_description}
												onChange={(e) => setEpisodeForm({...episodeForm, trigger_description: e.target.value})}
												required
											/>
										</div>
										<div className="form-group">
											<label className="form-label">Priority (1-5)</label>
											<select
												className="form-input form-select"
												value={episodeForm.priority}
												onChange={(e) => setEpisodeForm({...episodeForm, priority: parseInt(e.target.value)})}
											>
												<option value={1}>1 - Low</option>
												<option value={2}>2 - Normal</option>
												<option value={3}>3 - Medium</option>
												<option value={4}>4 - High</option>
												<option value={5}>5 - Critical</option>
											</select>
										</div>
										<div className="form-group">
											<label className="form-label">Member State Before</label>
											<textarea
												className="form-input form-textarea"
												value={episodeForm.member_state_before}
												onChange={(e) => setEpisodeForm({...episodeForm, member_state_before: e.target.value})}
												placeholder="Describe the member's state before this episode..."
											/>
										</div>
										<button type="submit" className="btn btn-primary">
											Create Episode
										</button>
									</form>
								)}
							</div>

							{/* Add Experiment */}
							<div className="card">
								<div className="card-header">
									<h3 className="card-title">Add New Experiment</h3>
									<button
										onClick={() => setShowExperimentForm(!showExperimentForm)}
										className="btn btn-primary btn-sm"
									>
										{showExperimentForm ? 'Cancel' : '+ Add Experiment'}
						</button>
					</div>
								{showExperimentForm && (
									<form onSubmit={addExperiment} className="space-y-4">
										<div className="form-group">
											<label className="form-label">Hypothesis</label>
											<textarea
												className="form-input form-textarea"
												value={experimentForm.hypothesis}
												onChange={(e) => setExperimentForm({...experimentForm, hypothesis: e.target.value})}
												placeholder="What do you want to test?"
												required
											/>
										</div>
										<div className="form-group">
											<label className="form-label">Template</label>
											<select
												className="form-input form-select"
												value={experimentForm.template}
												onChange={(e) => setExperimentForm({...experimentForm, template: e.target.value})}
											>
												<option value="">Custom Experiment</option>
												<option value="SLEEP_OPTIMIZATION">Sleep Optimization</option>
												<option value="NUTRITION_INTERVENTION">Nutrition Intervention</option>
												<option value="SUPPLEMENT_TRIAL">Supplement Trial</option>
												<option value="EXERCISE_PROTOCOL">Exercise Protocol</option>
												<option value="CGM_MEAL_TEST">CGM Meal Test</option>
											</select>
										</div>
										<div className="form-group">
											<label className="form-label">Member ID</label>
											<input
												type="text"
												className="form-input"
												value={experimentForm.member_id}
												onChange={(e) => setExperimentForm({...experimentForm, member_id: e.target.value})}
												required
											/>
										</div>
										<button type="submit" className="btn btn-primary">
											Create Experiment
										</button>
									</form>
								)}
				</div>

							{/* Data Overview */}
							<div className="card lg:col-span-2">
								<div className="card-header">
									<h3 className="card-title">Data Overview</h3>
									<button onClick={refreshAll} className="btn btn-sm">
										🔄 Refresh
						</button>
					</div>
								<div className="grid grid-cols-2 md:grid-cols-4 gap-4">
									<div className="text-center p-4 bg-blue-50 rounded-lg">
										<div className="text-2xl font-bold text-blue-600">{messages.length}</div>
										<div className="text-sm text-gray-600">Messages</div>
									</div>
									<div className="text-center p-4 bg-green-50 rounded-lg">
										<div className="text-2xl font-bold text-green-600">{suggestions.length}</div>
										<div className="text-sm text-gray-600">Suggestions</div>
									</div>
									<div className="text-center p-4 bg-yellow-50 rounded-lg">
										<div className="text-2xl font-bold text-yellow-600">{issues.length}</div>
										<div className="text-sm text-gray-600">Issues</div>
									</div>
									<div className="text-center p-4 bg-purple-50 rounded-lg">
										<div className="text-2xl font-bold text-purple-600">{episodes.length}</div>
										<div className="text-sm text-gray-600">Episodes</div>
									</div>
									<div className="text-center p-4 bg-red-50 rounded-lg">
										<div className="text-2xl font-bold text-red-600">{decisions.length}</div>
										<div className="text-sm text-gray-600">Decisions</div>
									</div>
									<div className="text-center p-4 bg-indigo-50 rounded-lg">
										<div className="text-2xl font-bold text-indigo-600">{experiments.length}</div>
										<div className="text-sm text-gray-600">Experiments</div>
									</div>
								</div>
				</div>
				{/* Trace Viewer */}
				<div className="card lg:col-span-2">
					<div className="card-header">
						<h3 className="card-title">Trace Viewer</h3>
					</div>
					<div className="space-y-3 max-h-96 overflow-y-auto">
						{traces.map((trace, i) => (
							<div key={i} className="p-4 border border-gray-200 rounded-lg">
								<div className="flex items-start justify-between mb-2">
									<h4 className="font-medium text-sm">{trace.message}</h4>
								</div>
								<pre className="text-xs bg-gray-100 p-2 rounded">
									{JSON.stringify(trace, null, 2)}
								</pre>
							</div>
						))}
						{traces.length === 0 && (
							<p className="text-sm text-gray-500 text-center py-8">No traces found.</p>
						)}
					</div>
				</div>
				</div>
				</div>
				)}

				{/* Analytics Tab */}
				{activeTab === "analytics" && (
					<div>
						<h2 className="text-2xl font-bold mb-6">Analytics & Insights</h2>
						
						<div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
							{/* Health Metrics */}
							<div className="card">
								<div className="card-header">
									<h3 className="card-title">Blood Sugar Trend</h3>
								</div>
								<ResponsiveContainer width="100%" height={300}>
									<LineChart data={healthMetrics}>
										<CartesianGrid strokeDasharray="3 3" />
										<XAxis dataKey="week" />
										<YAxis />
										<Tooltip />
										<Legend />
										<Line type="monotone" dataKey="health_metrics.blood_sugar_avg" stroke="#8884d8" name="Blood Sugar Avg" />
									</LineChart>
								</ResponsiveContainer>
							</div>

							<div className="card">
								<div className="card-header">
									<h3 className="card-title">A1C Improvement</h3>
								</div>
								<ResponsiveContainer width="100%" height={300}>
									<LineChart data={healthMetrics}>
										<CartesianGrid strokeDasharray="3 3" />
										<XAxis dataKey="week" />
										<YAxis />
										<Tooltip />
										<Legend />
										<Line type="monotone" dataKey="health_metrics.a1c" stroke="#82ca9d" name="A1C" />
									</LineChart>
								</ResponsiveContainer>
							</div>

							{/* Agent Performance */}
							<div className="card">
								<div className="card-header">
									<h3 className="card-title">Agent Performance</h3>
								</div>
								<div className="space-y-3">
									{Object.entries(agentPerformance).map(([agent, metrics]: [string, any]) => (
										<div key={agent} className="p-3 bg-gray-50 rounded-lg">
											<div className="flex items-center justify-between">
												<span className="font-medium">{agent}</span>
												<span className="text-sm text-gray-600">{metrics.suggestions_count} suggestions</span>
											</div>
											<div className="flex items-center justify-between mt-2 text-xs text-gray-500">
												<span>Avg. Response: {metrics.avg_response_time_ms.toFixed(0)}ms</span>
												<span>Satisfaction: {(metrics.satisfaction_score * 100).toFixed(0)}%</span>
											</div>
										</div>
									))}
								</div>
							</div>

							{/* Category Distribution */}
							<div className="card">
								<div className="card-header">
									<h3 className="card-title">Suggestion Categories</h3>
								</div>
								<div className="space-y-3">
									{['medical', 'nutrition', 'physio', 'logistics', 'performance', 'other'].map((category) => {
										const count = suggestions.filter(s => s.category === category).length;
										const percentage = suggestions.length > 0 ? (count / suggestions.length * 100).toFixed(1) : 0;
										return (
											<div key={category} className="flex items-center justify-between">
												<span className="font-medium capitalize">{category}</span>
												<div className="flex items-center gap-2">
													<div className="w-20 bg-gray-200 rounded-full h-2">
														<div
															className="bg-blue-500 h-2 rounded-full"
															style={{width: `${percentage}%`}}
														></div>
													</div>
													<span className="text-sm text-gray-600 w-12">{count}</span>
												</div>
											</div>
										);
									})}
								</div>
							</div>
							{/* Status Distribution */}
							<div className="card lg:col-span-2">
								<div className="card-header">
									<h3 className="card-title">Task Status Overview</h3>
						</div>
								<div className="grid grid-cols-2 md:grid-cols-5 gap-4">
									{(['proposed', 'accepted', 'in_progress', 'completed', 'dismissed'] as const).map((status) => {
										const count = suggestions.filter(s => s.status === status).length;
										return (
											<div key={status} className="text-center p-4 border rounded-lg">
												<div className="text-xl font-bold">{count}</div>
												<div className="text-sm text-gray-600 capitalize">{status.replace('_', ' ')}</div>
						</div>
										);
									})}
						</div>
					</div>
					</div>
				</div>
				)}
			</main>
		</div>
	);
}