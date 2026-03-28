export const metadata = {
  title: 'NexusGrid — AI Energy Intelligence',
  description: 'AI-Powered Energy Intelligence & Optimization Platform',
}

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  )
}
