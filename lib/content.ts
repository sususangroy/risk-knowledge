import fs from 'fs'
import path from 'path'
import matter from 'gray-matter'
import { remark } from 'remark'
import html from 'remark-html'
import remarkGfm from 'remark-gfm'
import { Article, Category, CATEGORIES } from './types'

export type { Article, Category }
export { CATEGORIES }

const contentDir = path.join(process.cwd(), 'content/knowledge')

export function getArticlesByCategory(category: Category): Article[] {
  const dir = path.join(contentDir, category)
  if (!fs.existsSync(dir)) return []
  const files = fs.readdirSync(dir).filter(f => f.endsWith('.md'))
  return files.map(filename => {
    const slug = `${category}/${filename.replace(/\.md$/, '')}`
    const raw = fs.readFileSync(path.join(dir, filename), 'utf8')
    const { data } = matter(raw)
    return {
      slug,
      title: data.title || filename,
      category,
      summary: data.summary || '',
      tags: data.tags || [],
    }
  })
}

export function getAllArticles(): Article[] {
  const categories: Category[] = ['basics', 'strategy', 'modeling', 'analytics']
  return categories.flatMap(getArticlesByCategory)
}

export async function getArticle(category: string, name: string): Promise<Article | null> {
  const filePath = path.join(contentDir, category, `${name}.md`)
  if (!fs.existsSync(filePath)) return null
  const raw = fs.readFileSync(filePath, 'utf8')
  const { data, content } = matter(raw)
  const processed = await remark().use(remarkGfm).use(html, { sanitize: false }).process(content)
  return {
    slug: `${category}/${name}`,
    title: data.title || name,
    category: category as Category,
    summary: data.summary || '',
    tags: data.tags || [],
    content: processed.toString(),
  }
}
