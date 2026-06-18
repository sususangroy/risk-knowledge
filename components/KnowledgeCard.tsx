import Link from 'next/link'
import { Article, CATEGORIES } from '@/lib/types'

export default function KnowledgeCard({ article }: { article: Article }) {
  const cat = CATEGORIES[article.category]
  const colorMap: Record<string, string> = {
    blue: 'bg-blue-50 text-blue-700',
    green: 'bg-green-50 text-green-700',
    purple: 'bg-purple-50 text-purple-700',
    orange: 'bg-orange-50 text-orange-700',
    red: 'bg-red-50 text-red-700',
  }
  return (
    <Link href={`/knowledge/${article.slug}`}>
      <div className="border border-gray-200 rounded-xl p-4 hover:shadow-md hover:border-blue-300 transition-all bg-white cursor-pointer">
        <div className="flex items-center gap-2 mb-2">
          <span className={`text-xs font-medium px-2 py-0.5 rounded-full ${colorMap[cat.color]}`}>
            {cat.label}
          </span>
        </div>
        <h3 className="font-semibold text-gray-800 mb-1">{article.title}</h3>
        <p className="text-sm text-gray-500 line-clamp-2">{article.summary}</p>
        {article.tags.length > 0 && (
          <div className="flex gap-1 mt-2 flex-wrap">
            {article.tags.map(tag => (
              <span key={tag} className="text-xs bg-gray-100 text-gray-500 px-2 py-0.5 rounded">
                {tag}
              </span>
            ))}
          </div>
        )}
      </div>
    </Link>
  )
}
