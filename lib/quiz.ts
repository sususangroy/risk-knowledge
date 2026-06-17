import fs from 'fs'
import path from 'path'
import { Question, Role, ROLES, DIFFICULTIES } from './types'

export type { Question, Role }
export { ROLES, DIFFICULTIES }

export function getQuestions(role: Role, difficulty?: string): Question[] {
  const filePath = path.join(process.cwd(), 'content/quiz', `${role}.json`)
  if (!fs.existsSync(filePath)) return []
  const data: Question[] = JSON.parse(fs.readFileSync(filePath, 'utf8'))
  if (difficulty && difficulty !== 'all') return data.filter(q => q.difficulty === difficulty)
  return data
}

export function getQuestionCount(role: Role): number {
  return getQuestions(role).length
}
