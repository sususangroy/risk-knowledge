import { notFound } from 'next/navigation'
import Link from 'next/link'
import { getArticle, CATEGORIES } from '@/lib/content'

export default async function ArticlePage({ params }: { params: Promise<{ slug: string[] }> }) {
  const { slug } = await params
  if (!slug || slug.length < 2) notFound()
  const [category, name] = slug
  const article = await getArticle(category, name)
  if (!article) notFound()
  const cat = CATEGORIES[article.category]

  return (
    <div className="max-w-3xl mx-auto">
      <div className="mb-6">
        <Link href="/knowledge" className="text-sm text-blue-600 hover:underline">← 返回知识库</Link>
      </div>
      <div className="flex items-center gap-2 mb-3">
        <span className="text-sm font-medium text-blue-700 bg-blue-50 px-2.5 py-0.5 rounded-full">
          {cat?.label}
        </span>
      </div>
      <h1 className="text-3xl font-bold text-gray-900 mb-4">{article.title}</h1>
      {article.summary && <p className="text-gray-500 text-base mb-6 border-b pb-6">{article.summary}</p>}
      {article.tags.length > 0 && (
        <div className="flex gap-2 mb-8 flex-wrap">
          {article.tags.map(tag => (
            <span key={tag} className="text-xs bg-gray-100 text-gray-500 px-2 py-1 rounded">{tag}</span>
          ))}
        </div>
      )}
      <div
        className="prose"
        dangerouslySetInnerHTML={{ __html: article.content || '' }}
      />
    </div>
  )
}
