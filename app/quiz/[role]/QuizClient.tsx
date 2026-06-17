'use client'
import { useState } from 'react'
import Link from 'next/link'
import QuizSession from '@/components/QuizSession'
import { Question, Difficulty, DIFFICULTIES } from '@/lib/types'

interface Props {
  role: string
  roleLabel: string
  allQuestions: Question[]
}

export default function QuizClient({ role, roleLabel, allQuestions }: Props) {
  const [difficulty, setDifficulty] = useState<Difficulty | 'all'>('all')

  const filtered = difficulty === 'all'
    ? allQuestions
    : allQuestions.filter(q => q.difficulty === difficulty)

  const difficultyOptions: Array<{ value: Difficulty | 'all'; label: string }> = [
    { value: 'all', label: '全部' },
    { value: 'junior', label: '初级' },
    { value: 'mid', label: '中级' },
    { value: 'senior', label: '高级' },
  ]

  return (
    <div>
      <div className="mb-6">
        <Link href="/quiz" className="text-sm text-blue-600 hover:underline">← 返回题库</Link>
      </div>
      <h1 className="text-2xl font-bold text-gray-900 mb-1">{roleLabel}</h1>
      <p className="text-gray-500 text-sm mb-6">共 {allQuestions.length} 道题</p>

      <div className="flex gap-2 mb-8">
        {difficultyOptions.map(opt => (
          <button
            key={opt.value}
            onClick={() => setDifficulty(opt.value)}
            className={`px-4 py-1.5 rounded-full text-sm font-medium transition-colors ${
              difficulty === opt.value
                ? 'bg-blue-700 text-white'
                : 'bg-gray-100 text-gray-600 hover:bg-gray-200'
            }`}
          >
            {opt.label}
            {opt.value !== 'all' && (
              <span className="ml-1 text-xs opacity-70">
                ({allQuestions.filter(q => q.difficulty === opt.value).length})
              </span>
            )}
          </button>
        ))}
      </div>

      <QuizSession questions={filtered} />
    </div>
  )
}
