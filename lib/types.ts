export type Category = 'basics' | 'strategy' | 'modeling' | 'analytics' | 'overseas'
export type Role = 'strategy' | 'modeling' | 'analytics'
export type Difficulty = 'junior' | 'mid' | 'senior'
export type QuestionType = 'single' | 'multi'

export interface Article {
  slug: string
  title: string
  category: Category
  summary: string
  tags: string[]
  content?: string
}

export interface Question {
  id: string
  question: string
  type: QuestionType
  difficulty: Difficulty
  options: string[]
  answer: string[]
  explanation: string
}

export const CATEGORIES: Record<Category, { label: string; description: string; color: string }> = {
  basics: { label: '风控基础', description: '信贷风控核心概念与术语', color: 'blue' },
  strategy: { label: '风控策略', description: '准入、额度、定价与贷后管理', color: 'green' },
  modeling: { label: '风控建模', description: '评分卡、机器学习模型与评估', color: 'purple' },
  analytics: { label: '数据分析', description: '风控数据分析方法与指标监控', color: 'orange' },
  overseas: { label: '海外市场', description: '东南亚与拉美信贷市场及监管政策', color: 'red' },
}

export const ROLES: Record<Role, { label: string; description: string }> = {
  strategy: { label: '风控策略', description: '准入规则、策略分析、业务逻辑' },
  modeling: { label: '风控建模', description: '模型开发、特征工程、模型评估' },
  analytics: { label: '风控数据分析', description: '数据分析、指标监控、报表洞察' },
}

export const DIFFICULTIES: Record<Difficulty, string> = {
  junior: '初级',
  mid: '中级',
  senior: '高级',
}
