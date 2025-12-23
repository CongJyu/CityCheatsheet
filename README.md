# CityCheatsheet

This is a cheatsheet auto-typesetting tool for students from City University of Hong Kong. This tool maps your Markdown notes to an A4 paper (both sides) automatically.

## Features

- **Auto Typesetting:** Automatically adjust font size and columns to fit your notes to a 2-page A4 cheatsheet.
- **Real-time A4 Preview:** You can preview your cheatsheet layout from the preview window.
- **Multiple Columns:** You can create several columns on one page.
- **Printable:** You can print your cheatsheet directly, or export it to a PDF file first.
- **Run It Locally:** The web-app can be run locally, without Internet connection, and **we do not and are not able to collect your information**. Suitable for those who have secret contents.

## Usages

1. **Paste your notes:** Copy your Markdown notes and paste it in the textbox.
2. **Typesetting:** Click "Auto Typesetting" button.
3. **Preview and export:** Click "Preview" and choose to print or export a PDF file. Or you can adjust the font size.

## Dev & Build

### Dependencies

```json
"dependencies": {
    "react": "^19.2.0",
    "react-dom": "^19.2.0",
    "marked": "^16.4.1",
    "dompurify": "^3.3.0",
    "pdfjs-dist": "^4.10.38",
    "jszip": "^3.10.1",
    "mammoth": "^1.7.2",
    "katex": "^0.16.9"
},
"devDependencies": {
    "@types/node": "^22.14.0",
    "@vitejs/plugin-react": "^5.0.0",
    "typescript": "~5.8.2",
    "vite": "^6.2.0"
}
```

### Install Dependencies

```shell
npm install
```

### Develop & Debug

```shell
npm run dev
```

This enables hot update when you change the code.

### Build

```shell
npm run build
```

And the build will generated in `dist/` directory.

### Preview

```shell
npm run preview
```

Preview local webpage after building the web-app.

## Thanks

This project is inspired from [jypengpeng/Cheatsheet-generator](https://github.com/jypengpeng/Cheatsheet-generator). Due to the situation that there is no LICENSE file decribed the project, it is not suitable to have a fork directly from the original project.

This rebuilt project removed some AI tools module, as we can access Google Gemini Pro from CityU Google Apps account, so it is not neccessay.
