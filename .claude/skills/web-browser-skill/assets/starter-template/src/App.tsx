function App() {
  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <header className="bg-white shadow-sm">
        <div className="max-w-7xl mx-auto px-4 py-4 sm:px-6 lg:px-8">
          <h1 className="text-2xl font-bold text-gray-900">
            Your Project Name
          </h1>
        </div>
      </header>

      {/* Main Content */}
      <main className="max-w-7xl mx-auto px-4 py-8 sm:px-6 lg:px-8">
        <div className="bg-white rounded-lg shadow p-6">
          <h2 className="text-xl font-semibold text-gray-800 mb-4">
            Welcome to Your Project
          </h2>
          <p className="text-gray-600 mb-4">
            Start building your frontend here. This template includes:
          </p>
          <ul className="list-disc list-inside text-gray-600 space-y-2">
            <li>React 18 with TypeScript</li>
            <li>Vite for fast development</li>
            <li>Tailwind CSS for styling</li>
            <li>Hot reload enabled</li>
          </ul>
          
          {/* Example Button */}
          <button 
            className="mt-6 px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors"
            onClick={() => alert('Button clicked!')}
          >
            Example Button
          </button>
        </div>
      </main>

      {/* Footer */}
      <footer className="bg-white border-t mt-auto">
        <div className="max-w-7xl mx-auto px-4 py-4 sm:px-6 lg:px-8">
          <p className="text-gray-500 text-sm text-center">
            Built with web-browser-skill
          </p>
        </div>
      </footer>
    </div>
  )
}

export default App
