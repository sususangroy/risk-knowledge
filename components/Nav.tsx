'use client'
import Link from 'next/link'
import { usePathname } from 'next/navigation'

export default function Nav() {
  const pathname = usePathname()
  const links = [
    { href: '/', label: '首页' },
    { href: '/knowledge', label: '知识库' },
    { href: '/quiz', label: '面试 Quiz' },
  ]
  return (
    <nav className="bg-white border-b border-gray-200 sticky top-0 z-50">
      <div className="max-w-5xl mx-auto px-4 flex items-center gap-8 h-14">
        <Link href="/" className="font-bold text-blue-700 text-lg tracking-tight">
          风控知识站
        </Link>
        <div className="flex gap-6">
          {links.map(l => (
            <Link
              key={l.href}
              href={l.href}
              className={`text-sm font-medium transition-colors ${
                pathname === l.href
                  ? 'text-blue-700'
                  : 'text-gray-600 hover:text-blue-700'
              }`}
            >
              {l.label}
            </Link>
          ))}
        </div>
      </div>
    </nav>
  )
}
