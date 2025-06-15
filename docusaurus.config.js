import {themes as prismThemes} from 'prism-react-renderer';

/** @type {import('@docusaurus/types').Config} */
const config = {
  title: 'Life with llm-d',
  tagline: 'A comprehensive guide to deploying, operating, and optimizing Large Language Model workloads using llm-d on Kubernetes and OpenShift',
  favicon: 'img/favicon.ico',

  url: 'https://jeremyeder.github.io',
  baseUrl: '/life-with-llm-d-book/',

  organizationName: 'jeremyeder',
  projectName: 'life-with-llm-d-book',

  onBrokenLinks: 'throw',
  onBrokenMarkdownLinks: 'warn',

  i18n: {
    defaultLocale: 'en',
    locales: ['en'],
  },

  markdown: {
    mermaid: true,
  },

  themes: ['@docusaurus/theme-mermaid'],

  presets: [
    [
      'classic',
      /** @type {import('@docusaurus/preset-classic').Options} */
      ({
        docs: {
          sidebarPath: './sidebars.js',
          editUrl: 'https://github.com/jeremyeder/life-with-llm-d-book/tree/main/',
          showLastUpdateTime: true,
          showLastUpdateAuthor: true,
        },
        blog: false,
        theme: {
          customCss: './src/css/custom.css',
        },
      }),
    ],
  ],

  themeConfig:
    /** @type {import('@docusaurus/preset-classic').ThemeConfig} */
    ({
      image: 'img/llm-d-social-card.jpg',
      navbar: {
        title: 'Life with llm-d',
        logo: {
          alt: 'llm-d Logo',
          src: 'img/logo.svg',
        },
        items: [
          {
            type: 'docSidebar',
            sidebarId: 'tutorialSidebar',
            position: 'left',
            label: 'Documentation',
          },
          {
            href: 'https://github.com/llm-d',
            label: 'llm-d GitHub',
            position: 'right',
          },
          {
            href: 'https://llm-d.ai',
            label: 'llm-d.ai',
            position: 'right',
          },
        ],
      },
      footer: {
        style: 'dark',
        links: [
          {
            title: 'Documentation',
            items: [
              {
                label: 'Getting Started',
                to: '/docs/01-introduction',
              },
            ],
          },
          {
            title: 'Community',
            items: [
              {
                label: 'llm-d Project',
                href: 'https://github.com/llm-d',
              },
              {
                label: 'Official Website',
                href: 'https://llm-d.ai',
              },
            ],
          },
          {
            title: 'More',
            items: [
              {
                label: 'Book Repository',
                href: 'https://github.com/jeremyeder/life-with-llm-d-book',
              },
            ],
          },
        ],
        copyright: `Copyright © ${new Date().getFullYear()} Life with llm-d. Built with Docusaurus.`,
      },
      prism: {
        theme: prismThemes.github,
        darkTheme: prismThemes.dracula,
        additionalLanguages: ['bash', 'yaml', 'json'],
      },
      colorMode: {
        defaultMode: 'dark',
        disableSwitch: false,
        respectPrefersColorScheme: true,
      },
    }),
};

export default config;
