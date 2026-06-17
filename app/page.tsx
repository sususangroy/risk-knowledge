import Link from 'next/link'
import { CATEGORIES, getArticlesByCategory } from '@/lib/content'
import { ROLES, getQuestionCount } from '@/lib/quiz'

export default function Home() {
  const categoryEntries = Object.entries(CATEGORIES)
  const colorBg: Record<string, string> = {
    blue: 'bg-blue-50 border-blue-200',
    green: 'bg-green-50 border-green-200',
    purple: 'bg-purple-50 border-purple-200',
    orange: 'bg-orange-50 border-orange-200',
  }
  const colorText: Record<string, string> = {
    blue: 'text-blue-700',
    green: 'text-green-700',
    purple: 'text-purple-700',
    orange: 'text-orange-700',
  }
  return (
    <div>
      <section className="text-center py-12 mb-10">
        <h1 className="text-4xl font-bold text-gray-900 mb-4">风控知识站</h1>
        <p className="text-lg text-gray-500 max-w-xl mx-auto">
          信贷风控领域知识普及 · 策略 / 建模 / 数据分析岗位面试题库
        </p>
        <div className="flex justify-center gap-4 mt-8">
          <Link href="/knowledge" className="bg-blue-700 text-white px-6 py-2.5 rounded-lg hover:bg-blue-800 transition-colors font-medium">
            浏览知识库
          </Link>
          <Link href="/quiz" className="border border-blue-700 text-blue-700 px-6 py-2.5 rounded-lg hover:bg-blue-50 transition-colors font-medium">
            开始刷题
          </Link>
        </div>
      </section>

      <section className="mb-12">
        <h2 className="text-xl font-bold text-gray-800 mb-5">知识分类</h2>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          {categoryEntries.map(([key, cat]) => {
            const articles = getArticlesByCategory(key as any)
            return (
              <Link key={key} href={`/knowledge?category=${key}`}>
                <div className={`border rounded-xl p-5 hover:shadow-md transition-all cursor-pointer ${colorBg[cat.color]}`}>
                  <h3 className={`font-bold text-lg mb-1 ${colorText[cat.color]}`}>{cat.label}</h3>
                  <p className="text-sm text-gray-500 mb-3">{cat.description}</p>
                  <p className="text-xs text-gray-400">{articles.length} 篇文章</p>
                </div>
              </Link>
            )
          })}
        </div>
      </section>

      <section>
        <h2 className="text-xl font-bold text-gray-800 mb-5">面试题库</h2>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          {Object.entries(ROLES).map(([key, role]) => {
            const count = getQuestionCount(key as any)
            return (
              <Link key={key} href={`/quiz/${key}`}>
                <div className="bg-white border border-gray-200 rounded-xl p-5 hover:shadow-md hover:border-blue-300 transition-all cursor-pointer">
                  <h3 className="font-bold text-gray-800 mb-1">{role.label}</h3>
                  <p className="text-sm text-gray-500 mb-3">{role.description}</p>
                  <p className="text-xs text-gray-400">{count} 道题</p>
                </div>
              </Link>
            )
          })}
        </div>
      </section>
    </div>
  )
}
