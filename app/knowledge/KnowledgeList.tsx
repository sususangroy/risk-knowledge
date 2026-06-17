'use client'
import { useSearchParams } from 'next/navigation'
import Link from 'next/link'
import { Article, Category, CATEGORIES } from '@/lib/types'
import KnowledgeCard from '@/components/KnowledgeCard'

interface Props {
  allArticles: Article[]
}

export default function KnowledgeList({ allArticles }: Props) {
  const searchParams = useSearchParams()
  const category = searchParams.get('category') as Category | null
  const categories = Object.keys(CATEGORIES) as Category[]
  const articles = category ? allArticles.filter(a => a.category === category) : allArticles

  return (
    <div>
      <h1 className="text-2xl font-bold text-gray-900 mb-6">知识库</h1>
      <div className="flex gap-2 mb-8 flex-wrap">
        <Link
          href="/knowledge"
          className={`px-4 py-1.5 rounded-full text-sm font-medium transition-colors ${
            !category ? 'bg-blue-700 text-white' : 'bg-gray-100 text-gray-600 hover:bg-gray-200'
          }`}
        >
          全部
        </Link>
        {categories.map(cat => (
          <Link
            key={cat}
            href={`/knowledge?category=${cat}`}
            className={`px-4 py-1.5 rounded-full text-sm font-medium transition-colors ${
              category === cat ? 'bg-blue-700 text-white' : 'bg-gray-100 text-gray-600 hover:bg-gray-200'
            }`}
          >
            {CATEGORIES[cat].label}
          </Link>
        ))}
      </div>
      {articles.length === 0 ? (
        <p className="text-gray-400 text-center py-16">暂无文章</p>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {articles.map(article => (
            <KnowledgeCard key={article.slug} article={article} />
          ))}
        </div>
      )}
    </div>
  )
}
