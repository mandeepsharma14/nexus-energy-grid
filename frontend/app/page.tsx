export default function Home() {
  return (
    <main style={{ 
      backgroundColor: '#0a0f1e', 
      minHeight: '100vh', 
      display: 'flex', 
      alignItems: 'center', 
      justifyContent: 'center',
      fontFamily: 'system-ui, sans-serif',
      color: '#e2e8f0'
    }}>
      <div style={{ textAlign: 'center' }}>
        <h1 style={{ fontSize: '3rem', color: '#38bdf8', marginBottom: '1rem' }}>
          ⚡ NexusGrid
        </h1>
        <p style={{ fontSize: '1.25rem', color: '#94a3b8', marginBottom: '2rem' }}>
          AI-Powered Energy Intelligence Platform
        </p>
        <p style={{ color: '#64748b' }}>
          © 2026 Mandeep Sharma. All rights reserved.
        </p>
      </div>
    </main>
  )
}
