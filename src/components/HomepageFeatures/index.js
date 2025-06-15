import clsx from 'clsx';
import styles from './styles.module.css';

const FeatureList = [
  {
    title: 'Data Scientist Focused',
    Svg: require('@site/static/img/undraw_data_scientist.svg').default,
    description: (
      <>
        Learn how to deploy and scale your large language models from 
        a data scientist perspective. From prototype to production with
        practical examples and real-world scenarios.
      </>
    ),
  },
  {
    title: 'SRE & Operations',
    Svg: require('@site/static/img/undraw_server.svg').default,
    description: (
      <>
        Master the operational aspects of llm-d deployments. Monitoring,
        scaling, troubleshooting, and maintaining production LLM workloads
        on Kubernetes and OpenShift.
      </>
    ),
  },
  {
    title: 'Production Ready',
    Svg: require('@site/static/img/undraw_kubernetes.svg').default,
    description: (
      <>
        Go beyond toy examples. Learn performance optimization, security,
        compliance, and the operational excellence needed for enterprise
        LLM deployments.
      </>
    ),
  },
];

function Feature({Svg, title, description}) {
  return (
    <div className={clsx('col col--4')}>
      <div className="text--center">
        <Svg className={styles.featureSvg} role="img" />
      </div>
      <div className="text--center padding-horiz--md">
        <h3>{title}</h3>
        <p>{description}</p>
      </div>
    </div>
  );
}

export default function HomepageFeatures() {
  return (
    <section className={styles.features}>
      <div className="container">
        <div className="row">
          {FeatureList.map((props, idx) => (
            <Feature key={idx} {...props} />
          ))}
        </div>
      </div>
    </section>
  );
}