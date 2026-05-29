# DeepSentinel

Research prototype for multimodal deepfake detection using audio-visual emotion analysis. Web UI built with Next.js.

## Tech Stack

- [Next.js 16](https://nextjs.org/) (App Router)
- React 19
- TypeScript
- Tailwind CSS 4
- [Radix UI](https://www.radix-ui.com/) + shadcn/ui components
- Recharts (charts), Lucide (icons)

## Prerequisites

- [Node.js](https://nodejs.org/) 18.18 or newer (20+ recommended)
- A package manager. This repo ships a `pnpm-lock.yaml`, so [pnpm](https://pnpm.io/) is recommended:
  ```bash
  npm install -g pnpm
  ```
  npm also works (a `package-lock.json` is included too).

## Setup

1. Clone the repo:
   ```bash
   git clone <repo-url>
   cd Thesis_MockUp
   ```

2. Install dependencies:
   ```bash
   pnpm install
   ```
   or
   ```bash
   npm install
   ```

## Run (development)

Start the dev server with hot reload:

```bash
pnpm dev
```
or `npm run dev`.

Open [http://localhost:3000](http://localhost:3000) in your browser.

## Build & run (production)

```bash
pnpm build   # compile production bundle
pnpm start   # serve the built app
```
or `npm run build` then `npm run start`.

## Other scripts

```bash
pnpm lint    # run ESLint
```

## Project structure

```
app/                 Next.js App Router (layout, page, global styles)
components/           React components
  deep-sentinel.tsx  Main app component
  ui/                shadcn/ui primitives
hooks/               Custom React hooks
lib/                 Utilities
public/              Static assets
```

## Notes

- TypeScript build errors are ignored at build time (`next.config.mjs`), so a failing type check will not block `pnpm build`.
- Image optimization is disabled (`images.unoptimized`).
