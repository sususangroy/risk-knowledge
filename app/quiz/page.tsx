import Link from 'next/link'
import { ROLES, Role } from '@/lib/quiz'

export default function QuizPage() {
  return (
    <div>
      <h1 className="text-2xl font-bold text-gray-900 mb-2">面试题库</h1>
      <p className="text-gray-500 mb-8">选择岗位方向，按难度刷题练习</p>
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        {Object.entries(ROLES).map(([key, role]) => (
          <Link key={key} href={`/quiz/${key}`}>
            <div className="bg-white border border-gray-200 rounded-xl p-6 hover:shadow-md hover:border-blue-300 transition-all cursor-pointer h-full">
              <h2 className="font-bold text-xl text-gray-800 mb-2">{role.label}</h2>
              <p className="text-gray-500 text-sm">{role.description}</p>
            </div>
          </Link>
        ))}
      </div>
    </div>
  )
}
