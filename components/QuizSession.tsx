'use client'
import { useState } from 'react'
import { Question } from '@/lib/types'
import DifficultyBadge from './DifficultyBadge'

export default function QuizSession({ questions }: { questions: Question[] }) {
  const [current, setCurrent] = useState(0)
  const [selected, setSelected] = useState<string[]>([])
  const [submitted, setSubmitted] = useState(false)
  const [score, setScore] = useState(0)
  const [finished, setFinished] = useState(false)

  if (questions.length === 0) {
    return <p className="text-gray-500 text-center py-12">暂无题目</p>
  }

  if (finished) {
    return (
      <div className="text-center py-16">
        <div className="text-5xl font-bold text-blue-700 mb-3">{score} / {questions.length}</div>
        <p className="text-gray-500 mb-8">答题完成！</p>
        <button
          onClick={() => { setCurrent(0); setSelected([]); setSubmitted(false); setScore(0); setFinished(false) }}
          className="bg-blue-700 text-white px-6 py-2 rounded-lg hover:bg-blue-800 transition-colors"
        >
          重新开始
        </button>
      </div>
    )
  }

  const q = questions[current]
  const isMulti = q.type === 'multi'

  const toggleOption = (opt: string) => {
    if (submitted) return
    if (isMulti) {
      setSelected(prev => prev.includes(opt) ? prev.filter(o => o !== opt) : [...prev, opt])
    } else {
      setSelected([opt])
    }
  }

  const isCorrect = (opt: string) => q.answer.includes(opt)

  const handleSubmit = () => {
    if (selected.length === 0) return
    setSubmitted(true)
    const correct =
      selected.length === q.answer.length && selected.every(s => q.answer.includes(s))
    if (correct) setScore(s => s + 1)
  }

  const handleNext = () => {
    if (current + 1 >= questions.length) {
      setFinished(true)
    } else {
      setCurrent(c => c + 1)
      setSelected([])
      setSubmitted(false)
    }
  }

  const optionStyle = (opt: string) => {
    const base = 'w-full text-left px-4 py-3 rounded-lg border transition-all text-sm '
    if (!submitted) {
      return base + (selected.includes(opt)
        ? 'border-blue-500 bg-blue-50 text-blue-800'
        : 'border-gray-200 hover:border-blue-300 bg-white')
    }
    if (isCorrect(opt)) return base + 'border-green-500 bg-green-50 text-green-800'
    if (selected.includes(opt) && !isCorrect(opt)) return base + 'border-red-400 bg-red-50 text-red-700'
    return base + 'border-gray-200 bg-white text-gray-400'
  }

  return (
    <div className="max-w-2xl mx-auto">
      <div className="flex items-center justify-between mb-6 text-sm text-gray-500">
        <span>{current + 1} / {questions.length}</span>
        <DifficultyBadge difficulty={q.difficulty} />
      </div>

      <div className="bg-white border border-gray-200 rounded-xl p-6 mb-4">
        <div className="flex items-center gap-2 mb-4">
          <span className="text-xs bg-gray-100 text-gray-500 px-2 py-0.5 rounded">
            {isMulti ? '多选题' : '单选题'}
          </span>
        </div>
        <p className="text-gray-800 font-medium text-base leading-relaxed mb-6">{q.question}</p>
        <div className="flex flex-col gap-2">
          {q.options.map(opt => (
            <button key={opt} onClick={() => toggleOption(opt)} className={optionStyle(opt)}>
              {opt}
            </button>
          ))}
        </div>
      </div>

      {submitted && (
        <div className="bg-blue-50 border border-blue-200 rounded-xl p-4 mb-4 text-sm text-gray-700">
          <p className="font-semibold text-blue-700 mb-1">解析</p>
          <p>{q.explanation}</p>
        </div>
      )}

      <div className="flex justify-end">
        {!submitted ? (
          <button
            onClick={handleSubmit}
            disabled={selected.length === 0}
            className="bg-blue-700 text-white px-6 py-2 rounded-lg hover:bg-blue-800 disabled:opacity-40 transition-colors"
          >
            提交
          </button>
        ) : (
          <button onClick={handleNext} className="bg-blue-700 text-white px-6 py-2 rounded-lg hover:bg-blue-800 transition-colors">
            {current + 1 >= questions.length ? '查看结果' : '下一题'}
          </button>
        )}
      </div>
    </div>
  )
}
