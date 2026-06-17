import { Difficulty, DIFFICULTIES } from '@/lib/types'

const colorMap: Record<Difficulty, string> = {
  junior: 'bg-green-100 text-green-700',
  mid: 'bg-yellow-100 text-yellow-700',
  senior: 'bg-red-100 text-red-700',
}

export default function DifficultyBadge({ difficulty }: { difficulty: Difficulty }) {
  return (
    <span className={`text-xs font-medium px-2 py-0.5 rounded-full ${colorMap[difficulty]}`}>
      {DIFFICULTIES[difficulty]}
    </span>
  )
}
