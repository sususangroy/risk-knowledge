import { Suspense } from 'react'
import { getAllArticles } from '@/lib/content'
import KnowledgeList from './KnowledgeList'

export default function KnowledgePage() {
  const allArticles = getAllArticles()
  return (
    <Suspense fallback={<div className="text-gray-400 text-center py-16">加载中...</div>}>
      <KnowledgeList allArticles={allArticles} />
    </Suspense>
  )
}
