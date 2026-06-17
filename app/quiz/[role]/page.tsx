import { notFound } from 'next/navigation'
import { ROLES, getQuestions, Role } from '@/lib/quiz'
import QuizClient from './QuizClient'

export default async function QuizRolePage({ params }: { params: Promise<{ role: string }> }) {
  const { role } = await params
  if (!(role in ROLES)) notFound()
  const questions = getQuestions(role as Role)
  const roleLabel = ROLES[role as Role].label
  return <QuizClient role={role} roleLabel={roleLabel} allQuestions={questions} />
}
