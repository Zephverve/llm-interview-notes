# 网站构建与部署

这个目录把根目录的 Markdown 同步到 `docs/`，再用 MkDocs Material 生成静态站点。

## 本地预览

```powershell
python 网站/build.py --serve
```

- 电脑访问：`http://127.0.0.1:8000/`
- 手机访问：确保手机和电脑在同一 Wi-Fi，打开终端里打印的 `http://<局域网 IP>:8000/`
- 修改根目录的 Markdown 后会自动同步并重建；按 `Ctrl+C` 停止。

## 只构建

```powershell
python 网站/build.py
```

构建产物在 `网站/site/`，可以直接上传到任意静态托管平台。

## 目录说明

| 路径 | 用途 |
| --- | --- |
| `网站/mkdocs.yml` | 站点配置、导航、主题、Markdown 扩展 |
| `网站/index.template.md` | 首页模板，引用根目录 `README.md` |
| `网站/assets/` | 样式、脚本、Logo、本地 MathJax |
| `网站/build.py` | 同步内容并调用 MkDocs |
| `网站/docs/` | 自动生成的 MkDocs 源目录，不要手改 |
| `网站/site/` | 自动生成的静态站点，不要手改 |

## GitHub Pages

仓库根目录已包含 `.github/workflows/deploy.yml`。推送到 `main` 后：

1. 打开仓库的 **Settings → Pages**。
2. 在 **Build and deployment → Source** 选择 **GitHub Actions**。
3. 等待 Actions 完成，站点地址为 `https://<用户名>.github.io/<仓库名>/`。

如果换了用户名或仓库名，需要同步修改 `网站/mkdocs.yml` 里的 `site_url`。

## Cloudflare Pages

在 Cloudflare Pages 连接 Git 仓库后使用：

- **Build command**: `python 网站/build.py`
- **Build output directory**: `网站/site`
- **Root directory**: 仓库根目录

如果仓库根目录是上一级的 `默认项目`，把命令改成 `python 大模型面试资料/网站/build.py`，输出目录改成 `大模型面试资料/网站/site`。

## 自定义域名与 HTTPS

- GitHub Pages：在 **Settings → Pages → Custom domain** 填写域名，并按提示配置 DNS。
- Cloudflare Pages：在 **Custom domains** 添加域名，Cloudflare 会自动签发 HTTPS 证书。
- 如果使用中国大陆服务器托管，需要先完成 ICP 备案；GitHub Pages / Cloudflare Pages 的默认子域名不需要备案。

## 已验证的移动端要点

- 主题响应式布局，手机端导航为抽屉式。
- 表格、代码块、行内公式和独立公式可横向滚动。
- MathJax 使用本地文件，不依赖 Google Fonts 或境外 CDN。
- 中文搜索基于 MkDocs Material 的 `zh` 分词配置。
