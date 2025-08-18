"use client";

import React, { useEffect, useMemo, useRef, useState } from "react";

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

	const [simulationResults, setSimulationResults] = useState<any>(null);
	const [xmlInput, setXmlInput] = useState("");

	const runChatSimulation = async (e: React.FormEvent) => {
		e.preventDefault();
		if (!xmlInput.trim()) {
			alert("Please paste the XML content before running the simulation.");
			return;
		}

		setIsLoading(true);
		try {
			const response = await fetch(`${backendBase}/simulation/run`, {
				method: "POST",
				headers: { "Content-Type": "application/json" },
				body: JSON.stringify({ xml_content: xmlInput }),
			});

			if (response.ok) {
				const results = await response.json();
				setSimulationResults(results);
			}
		} catch (error) {
			console.error("Error running chat simulation:", error);
		} finally {
			setIsLoading(false);
		}
	};

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
			const [messagesRes, suggestionsRes, issuesRes, episodesRes, decisionsRes, experimentsRes] = await Promise.all([
				fetch(`${backendBase}/messages`),
				fetch(`${backendBase}/suggestions`),
				fetch(`${backendBase}/issues`),
				fetch(`${backendBase}/episodes`),
				fetch(`${backendBase}/decisions`),
				fetch(`${backendBase}/experiments`)
			]);

			if (messagesRes.ok) {
				const data = await messagesRes.json();
				// Extract messages array from the response object
				const messagesArray = data.messages || data || [];
				setMessages(messagesArray as Message[]);
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
		} catch (error) {
			console.error('Error loading data:', error);
			// Ensure messages is always an array even on error
			setMessages([]);
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
				// Extract messages array from the response object
				const messagesArray = newMessages.messages || newMessages || [];
				setMessages(messagesArray as Message[]);
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
		totalMessages: Array.isArray(messages) ? messages.length : 0,
		pendingSuggestions: Array.isArray(pendingSuggestions) ? pendingSuggestions.length : 0,
		activeTasks: Array.isArray(planItems.in_progress) ? planItems.in_progress.length : 0,
		completedTasks: Array.isArray(planItems.completed) ? planItems.completed.length : 0,
		openIssues: Array.isArray(issues) ? issues.filter(i => i.status !== 'resolved').length : 0,
		activeExperiments: Array.isArray(experiments) ? experiments.filter(e => e.status === 'running').length : 0,
    };

	const exportToCsv = (data: any[], filename: string) => {
		const headers = ['content', 'date', 'time', 'sender'];
		const csvRows = [headers.join(',')];

		for (const row of data) {
			const values = headers.map(header => {
				const value = header === 'content' ? row['message'] : row[header];
				const escaped = ('' + value).replace(/"/g, '\\"');
				return `"${escaped}"`;
			});
			csvRows.push(values.join(','));
		}

		const blob = new Blob([csvRows.join('\\n')], { type: 'text/csv' });
		const url = window.URL.createObjectURL(blob);
		const a = document.createElement('a');
		a.setAttribute('hidden', '');
		a.setAttribute('href', url);
		a.setAttribute('download', filename);
		document.body.appendChild(a);
		a.click();
		document.body.removeChild(a);
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
						 simulating Chat
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
										{!Array.isArray(messages) && (
											<div className="message agent">
												<div className="message-header">System</div>
												<div className="message-content">
													<div className="loading"></div> Loading messages...
												</div>
											</div>
										)}
										{Array.isArray(messages) && messages.length === 0 && (
											<div className="message agent">
												<div className="message-header">AI Team</div>
												<div className="message-content">
													No messages yet. Start a conversation!
												</div>
											</div>
										)}
										{Array.isArray(messages) && messages.map((msg, idx) => (
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
									{Array.isArray(pendingSuggestions) && pendingSuggestions.map((suggestion) => (
										<div key={suggestion.id} className="p-4 border border-gray-200 rounded-lg">
											<div className="flex items-start justify-between mb-2">
												<h4 className="text-sm font-medium">{suggestion.title}</h4>
												<span className="text-xs text-gray-500">{suggestion.agent}</span>
											</div>
											<p className="text-xs text-gray-600 mb-3">{suggestion.details}</p>
											<div className="flex items-start justify-between">
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
									{(!Array.isArray(pendingSuggestions) || pendingSuggestions.length === 0) && (
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
								<div className="stat-value text-blue-500">{Array.isArray(planItems.todo) ? planItems.todo.length : 0}</div>
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
												{status === 'todo' ? `To-Do (${Array.isArray(planItems.todo) ? planItems.todo.length : 0})` : 
												 status === 'in_progress' ? `In Progress (${Array.isArray(planItems.in_progress) ? planItems.in_progress.length : 0})` :
												 `Completed (${Array.isArray(planItems.completed) ? planItems.completed.length : 0})`}
											</button>
										))}
									</div>
								</div>
								<div className="space-y-3 max-h-96 overflow-y-auto">
									{Array.isArray(planItems[planFilter]) && planItems[planFilter].map((item) => (
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
									{(!Array.isArray(planItems[planFilter]) || planItems[planFilter].length === 0) && (
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
									{Array.isArray(issues) && issues.filter(i => (i.status || 'open') !== 'resolved').map((issue) => (
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
									{(!Array.isArray(issues) || issues.length === 0) && (
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
									{Array.isArray(episodes) && episodes.map((episode) => (
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
									{Array.isArray(experiments) && experiments.map((experiment) => (
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
									{(!Array.isArray(experiments) || experiments.length === 0) && (
										<p className="text-sm text-gray-500 text-center py-8">No experiments running.</p>
									)}
								</div>
							</div>
						</div>
						</div>
				)}

				{/* Chat Simulation Tab */}
				{activeTab === "data" && (
					<div>
						<h2 className="text-2xl font-bold mb-6">Chat Simulation</h2>
						<div className="card">
							<div className="card-header">
								<h3 className="card-title">Journey Description</h3>
								<p className="card-subtitle">
									Enter the month-wise journey description for Rohan Patel.
									Include episodes, major messages, and test/report data.
								</p>
							</div>
							<form onSubmit={runChatSimulation}>
								<div className="form-group">
									<textarea
										className="form-input form-textarea h-64"
										value={xmlInput}
										onChange={(e) => setXmlInput(e.target.value)}
										placeholder="Paste the content of episodes.xml here..."
									/>
								</div>
								<button type="submit" className="btn btn-primary" disabled={isLoading}>
									{isLoading ? <div className="loading"></div> : "Run Simulation"}
								</button>
							</form>

							{simulationResults && (
								<div className="mt-6">
									<h3 className="text-xl font-bold mb-4">Simulation Results</h3>

									{/* Journey Summary */}
									<div className="mb-6">
										<h4 className="text-lg font-semibold mb-2">Journey Summary</h4>
										<div className="card bg-gray-50">
											<div className="card-body">
												<h5 className="card-title text-primary">Key Milestones</h5>
												<ul className="list-disc pl-5 space-y-2 mb-4">
													{Array.isArray(simulationResults.journey_summary?.key_milestones) && simulationResults.journey_summary.key_milestones.map((milestone: string, idx: number) => (
														<li key={idx} className="text-gray-700">{milestone}</li>
													))}
												</ul>
												<h5 className="card-title text-primary">Overall Progress</h5>
												<p className="text-gray-700">{simulationResults.journey_summary.overall_progress}</p>
											</div>
										</div>
									</div>

									{/* Conversation History */}
									<div className="mb-6">
										<h4 className="text-lg font-semibold mb-2">Conversation History</h4>
										<div className="chat-container">
											<div className="chat-messages">
												{Array.isArray(simulationResults.conversation_history) && simulationResults.conversation_history.map((msg: Message, idx: number) => (
													<div key={idx} className={`message ${msg.sender === 'Rohan' ? 'user' : 'agent'}`}>
														<div className="message-header">
															{msg.sender}
														</div>
														<div className="message-content">
															{msg.message}
														</div>
													</div>
												))}
											</div>
										</div>
										<button onClick={() => exportToCsv(simulationResults.conversation_history, 'conversation_history.csv')} className="btn btn-secondary mt-2">
											Download Conversation CSV
										</button>
									</div>
								</div>
							)}
						</div>
					</div>
				)}

				{/* Analytics Tab */}
				{activeTab === "analytics" && (
					<div>
						<h2 className="text-2xl font-bold mb-6">Analytics & Insights</h2>
						
						<div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
							{/* Agent Performance */}
							<div className="card">
								<div className="card-header">
									<h3 className="card-title">Agent Activity</h3>
								</div>
								<div className="space-y-3">
									{['Ruby', 'Dr. Warren', 'Advik', 'Carla', 'Rachel', 'Neel'].map((agent) => {
										const agentSuggestions = suggestions.filter(s => s.agent === agent).length;
										return (
											<div key={agent} className="flex items-center justify-between p-3 bg-gray-50 rounded-lg">
												<span className="font-medium">{agent}</span>
												<span className="text-sm text-gray-600">{agentSuggestions} suggestions</span>
											</div>
										);
									})}
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