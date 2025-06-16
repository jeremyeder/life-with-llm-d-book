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
          src: 'img/llm-d-icon.png',
        },
        items: [
          {
            type: 'docSidebar',
            sidebarId: 'tutorialSidebar',
            position: 'left',
            label: 'Book Guide',
          },
          {
            to: '/docs/introduction',
            position: 'left',
            label: 'Introduction',
          },
          {
            href: 'https://llm-d.ai',
            label: 'What is llm-d?',
            position: 'left',
          },
          {
            href: 'https://github.com/llm-d',
            className: 'header-github-link',
            'aria-label': 'GitHub repository',
            position: 'right',
          },
        ],
      },
      footer: {
        style: 'dark',
        links: [
          {
            title: 'Book Guide',
            items: [
              {
                label: 'Getting Started',
                to: '/docs/introduction',
              },
              {
                label: 'Data Scientist Workflows',
                to: '/docs/data-scientist-workflows',
              },
              {
                label: 'SRE Operations',
                to: '/docs/sre-operations',
              },
            ],
          },
          {
            title: 'Architecture',
            items: [
              {
                label: 'Understanding Architecture',
                to: '/docs/understanding-architecture',
              },
              {
                label: 'Performance Optimization',
                to: '/docs/performance-optimization',
              },
              {
                label: 'Security & Compliance',
                to: '/docs/security-compliance',
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
              {
                label: 'Slack Community',
                href: 'https://llm-d.ai/slack',
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
              {
                label: 'News & Updates',
                href: 'https://llm-d.ai/news',
              },
            ],
          },
        ],
        copyright: `Copyright © ${new Date().getFullYear()} Life with llm-d. Built with Docusaurus.`,
      },
      prism: {
        theme: prismThemes.vsLight,
        darkTheme: prismThemes.vsDark,
        additionalLanguages: ['bash', 'yaml', 'json', 'python', 'docker'],
      },
      colorMode: {
        defaultMode: 'light',
        disableSwitch: false,
        respectPrefersColorScheme: true,
      },
    }),
};

export default config;
