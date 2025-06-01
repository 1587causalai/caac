# 📚 文档网站部署指南

本指南详细介绍了 CAAC 项目文档网站的构建、部署和维护流程。

## 🎯 网站架构

### 技术栈
- **框架**: [Docsify](https://docsify.js.org/) - 轻量级文档网站生成器
- **托管**: GitHub Pages - 免费静态网站托管
- **自动化**: GitHub Actions - CI/CD 自动部署
- **样式**: 自定义CSS + Vue主题

### 网站结构
```
docs/
├── 📄 index.html              # Docsify 配置和入口
├── 📄 README.md               # 首页内容
├── 📄 _sidebar.md             # 侧边栏导航
├── 📄 _coverpage.md           # 封面页
├── 
├── 📁 tutorials/              # 使用教程
│   ├── quickstart.md          # 快速开始
│   ├── installation.md        # 安装指南
│   └── faq.md                 # 常见问题
├── 
├── 📁 theory/                 # 理论基础
│   ├── motivation.md          # 项目动机
│   ├── mathematical_foundations.md # 数学基础
│   └── design_principles.md   # 设计原则
├── 
├── 📁 api/                    # API文档
├── 📁 experiments/            # 实验结果
├── 📁 development/            # 开发文档
├── 📁 assets/                 # 静态资源
└── 📁 appendix/               # 附录
```

## 🚀 自动部署流程

### GitHub Actions 工作流

文档网站通过 `.github/workflows/docs.yml` 实现自动部署：

```yaml
# 触发条件
on:
  push:
    branches: [ main, master ]
    paths: [ 'docs/**' ]
  pull_request:
    branches: [ main, master ]
    paths: [ 'docs/**' ]
```

### 部署步骤

1. **文档验证** 🔧
   - 检查必需文件
   - 验证内部链接
   - 确保文档结构完整

2. **构建网站** 🏗️
   - 安装 Docsify CLI
   - 处理 Markdown 文件
   - 生成静态网站

3. **自动部署** 🚀
   - 推送到 `gh-pages` 分支
   - 更新 GitHub Pages
   - 生成部署报告

4. **质量检查** 📊
   - 统计文档数量和大小
   - 生成分析报告
   - PR 预览评论

## 🛠️ 本地开发

### 环境准备

```bash
# 安装 Node.js 和 npm
# 访问 https://nodejs.org 下载

# 安装 Docsify CLI
npm install -g docsify-cli

# 克隆项目
git clone https://github.com/1587causalai/caac.git
cd caac_project
```

### 本地预览

```bash
# 启动文档服务器
cd docs
docsify serve

# 或从项目根目录
docsify serve docs

# 访问 http://localhost:3000
```

### 开发工作流

```bash
# 1. 创建分支
git checkout -b feature/docs-update

# 2. 编辑文档
# 修改 docs/ 目录下的 Markdown 文件

# 3. 本地预览
docsify serve docs

# 4. 提交更改
git add docs/
git commit -m "📚 Update documentation"

# 5. 推送分支
git push origin feature/docs-update

# 6. 创建 Pull Request
# GitHub Actions 会自动验证和预览
```

## 📝 内容编写规范

### Markdown 规范

```markdown
# 🎯 页面标题

> 📋 简要描述或引言

## 🔧 一级标题

### 二级标题

- 使用 emoji 增强可读性
- 统一的代码块格式
- 清晰的表格结构

**重要提示:**
- 使用 `**粗体**` 强调重点
- 使用 `代码块` 标记代码
- 使用 > 引用块标记重要信息

```bash
# 代码示例
python run_experiments.py --quick
```
```

### 文档组织原则

1. **层次清晰** - 使用合理的标题层级
2. **内容完整** - 每个主题都有详细说明
3. **例子丰富** - 提供充足的代码示例
4. **交叉引用** - 合理使用内部链接
5. **持续更新** - 与代码变更同步

### 样式指南

| 元素 | 格式 | 示例 |
|------|------|------|
| 文件路径 | `代码格式` | `src/models/caac_ovr_model.py` |
| 命令 | `代码格式` | `python run_experiments.py --quick` |
| 参数 | **粗体** | **representation_dim** |
| 强调 | emoji + 粗体 | 🎯 **核心功能** |

## 🔧 网站配置

### Docsify 配置

`docs/index.html` 中的主要配置项：

```javascript
window.$docsify = {
  name: 'CAAC',                    // 项目名称
  repo: 'https://github.com/...',  // GitHub 仓库
  loadSidebar: true,               // 启用侧边栏
  coverpage: true,                 // 启用封面页
  search: {...},                   // 搜索配置
  copyCode: {...},                 // 代码复制
  pagination: {...}                // 分页导航
}
```

### 自定义样式

主要的 CSS 自定义：

```css
:root {
  --base-color: #2c3e50;      /* 主色调 */
  --theme-color: #3498db;     /* 主题色 */
  --sidebar-width: 300px;     /* 侧边栏宽度 */
}

.cover.show {
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
}
```

### 插件配置

启用的 Docsify 插件：

- **搜索** - 全文搜索功能
- **代码复制** - 一键复制代码块
- **图片缩放** - 点击放大图片
- **分页导航** - 上一页/下一页
- **数学公式** - KaTeX 数学渲染
- **Tab 标签** - 内容分组显示

## 📊 网站维护

### 内容更新流程

1. **定期审查** 📋
   - 每月检查内容准确性
   - 更新过时的信息
   - 修复失效链接

2. **版本同步** 🔄
   - 代码更新时同步文档
   - 新功能添加说明文档
   - API 变更更新文档

3. **用户反馈** 💬
   - 处理文档相关 Issue
   - 根据用户建议改进
   - 添加常见问题解答

### 监控和分析

**GitHub Actions 报告**:
- 📊 文档构建状态
- 🔗 链接验证结果
- 📈 内容统计分析

**用户体验优化**:
- 💻 响应式设计测试
- 🔍 搜索功能效果
- ⚡ 页面加载速度

### 备份和恢复

```bash
# 文档备份
git clone https://github.com/1587causalai/caac.git caac-backup
cd caac-backup
git checkout gh-pages  # GitHub Pages 分支

# 恢复部署
git checkout main
git push origin main    # 触发自动部署
```

## 🌐 域名和 SSL

### 自定义域名配置

1. **DNS 设置**:
   ```
   CNAME: docs.yourdomain.com -> username.github.io
   ```

2. **GitHub Pages 配置**:
   - 在仓库设置中添加自定义域名
   - 启用 HTTPS 强制重定向

3. **工作流更新**:
   ```yaml
   with:
     cname: docs.yourdomain.com  # 在 docs.yml 中设置
   ```

### SSL 证书

GitHub Pages 自动提供 Let's Encrypt SSL 证书：
- ✅ 自动续期
- ✅ 支持自定义域名
- ✅ 强制 HTTPS

## 🚨 故障排除

### 常见问题

**部署失败**:
```bash
# 检查工作流状态
# 访问 GitHub Actions 页面查看详细日志

# 本地验证
docsify serve docs
```

**链接失效**:
```bash
# 运行链接检查
find docs -name "*.md" -exec grep -l "](.*\.md)" {} \;
```

**样式问题**:
```bash
# 清除浏览器缓存
# 检查 CSS 文件语法
# 验证 Docsify 版本兼容性
```

### 紧急修复

```bash
# 快速修复流程
git checkout main
git pull origin main

# 编辑问题文件
vim docs/problematic-file.md

# 快速部署
git add docs/
git commit -m "🚨 Hotfix documentation"
git push origin main

# 等待自动部署（约 2-5 分钟）
```

## 📞 获取帮助

- 📖 **Docsify 文档**: https://docsify.js.org/
- 🛠️ **GitHub Pages**: https://docs.github.com/pages
- 💬 **项目 Issue**: 在 GitHub 仓库中提交问题

---

🎉 **文档网站现已完全自动化！** 只需编辑 Markdown 文件，其余交给 CI/CD 处理。

> 💡 **最佳实践**: 小量多次更新，每次更改都有清晰的提交信息，利用 PR 进行代码审查。 