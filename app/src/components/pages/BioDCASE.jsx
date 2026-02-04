import { useState, useEffect } from 'react';

const teamMembers = [
  {
    id: 1,
    name: 'Ben McEwen',
    affiliation: 'Postdoctoral researcher, Tilburg University, Netherlands',
    description: 'Ben is a Postdoctoral Researcher applying active learning for biodiversity monitoring at a transnational scale (from Norway to Spain) through the TABMON project. Previously, Ben researched active learning methods for at-risk and invasive species detection',
    contact: 'benmcewen@outlook.com',
    googleScholar: 'https://scholar.google.com/citations?hl=en&user=x47JZUkAAAAJ&view_op=list_works&sortby=pubdate',
    website: 'https://www.benmcewen-phd.com/',
    img: '../../../dist/profiles/ben_mcewen.jpg'
  },
  {
    id: 2,
    name: 'Lukas Rauch',
    affiliation: 'PhD Candidate, Kassel University, Germany',
    description: 'Brief bio and research interests go here. This section can be expanded to show more details about the team member\'s background and contributions to the project.',
    googleScholar: 'https://scholar.google.com/citations?hl=en&user=bB2A6e0AAAAJ&view_op=list_works&sortby=pubdate',
    // website: 'https://example.com/',
    img: '../../../dist/profiles/lukas_rauch.jpg'
  },
  {
    id: 3,
    name: 'Marek Herde',
    affiliation: 'PhD Candidate, Kassel University, Germany',
    description: 'Brief bio and research interests go here. This section can be expanded to show more details about the team member\'s background and contributions to the project.',
    googleScholar: 'https://scholar.google.com/citations?hl=en&user=pwRDfMQAAAAJ&view_op=list_works&sortby=pubdate',
    // website: 'https://example.com/',
    img: '../../../dist/profiles/marek_herde.JPG'
  },
  {
    id: 4,
    name: 'Shiqi Zhang',
    affiliation: 'PhD Candidate, Tampere University, Finland',
    description: 'Brief bio and research interests go here. This section can be expanded to show more details about the team member\'s background and contributions to the project.',
    googleScholar: 'https://scholar.google.com/citations?hl=en&user=fnOCg-8AAAAJ&view_op=list_works&sortby=pubdate',
    // website: 'https://example.com/',
    img: '../../../dist/profiles/shiqi_zhang.jpg'
  },
  {
    id: 5,
    name: 'Rupa Kurinchi-Vendhan',
    affiliation: 'PhD Candidate, Massachusetts Institute of Technology',
    description: 'Brief bio and research interests go here. This section can be expanded to show more details about the team member\'s background and contributions to the project.',
    googleScholar: 'https://scholar.google.com/citations?hl=en&user=YY9cf7sAAAAJ&view_op=list_works&sortby=pubdate',
    website: 'https://rupakv.com/',
    img: '../../../dist/profiles/rupa_kurinchi_vendhan.jfif'
  },
  {
    id: 6,
    name: 'John Martinsson',
    affiliation: 'RISE Research Institute of Sweden',
    description: 'John is a PhD candidate at Lund University and Research Scientist at RISE Research Institutes of Sweden, developing annotation-efficient machine listening methods for bioacoustics. His research directly addresses the bottleneck of labeling costs through active learning.',
    googleScholar: 'https://scholar.google.com/citations?hl=en&user=sAMIwlMAAAAJ&view_op=list_works&sortby=pubdate',
    website: 'https://johnmartinsson.org/',
    img: '../../../dist/profiles/john_martinsson.jpg'
  },
  {
    id: 7,
    name: 'Sara Beery',
    affiliation: 'Associate Professor, Massachusetts Institute of Technology',
    description: 'Brief bio and research interests go here. This section can be expanded to show more details about the team member\'s background and contributions to the project.',
    googleScholar: 'https://scholar.google.com/citations?user=Hbr4c10AAAAJ&hl=en&oi=ao',
    website: 'https://beerys.github.io/',
    img: '../../../dist/profiles/sara_beery.jpeg'
  },
];

function ScholarIcon() {
  return (
    <svg width="20" height="20" viewBox="0 0 24 24" fill="currentColor">
      <path d="M12 24a7 7 0 1 1 0-14 7 7 0 0 1 0 14zm0-24L0 9.5l4.838 3.94A8 8 0 0 1 12 9a8 8 0 0 1 7.162 4.44L24 9.5z"/>
    </svg>
  );
}

function WebsiteIcon() {
  return (
    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <circle cx="12" cy="12" r="10"/>
      <line x1="2" y1="12" x2="22" y2="12"/>
      <path d="M12 2a15.3 15.3 0 0 1 4 10 15.3 15.3 0 0 1-4 10 15.3 15.3 0 0 1-4-10 15.3 15.3 0 0 1 4-10z"/>
    </svg>
  );
}

function TeamMemberCard({ member }) {
  const [expanded, setExpanded] = useState(false);
  const [hovered, setHovered] = useState(false);

  return (
    <div
      style={{
        backgroundColor: 'rgba(255, 255, 255, 0.1)',
        borderRadius: '12px',
        padding: '20px',
        textAlign: 'center',
        cursor: 'pointer',
        transition: 'all 0.3s ease',
        flex: '0 1 calc(50% - 10px)',
        minWidth: '280px',
        boxSizing: 'border-box',
        border: hovered ? '2px solid rgba(255, 255, 255, 0.5)' : '2px solid transparent',
        transform: hovered ? 'scale(1.02)' : 'scale(1)',
      }}
      onMouseEnter={() => setHovered(true)}
      onMouseLeave={() => setHovered(false)}
      onClick={() => setExpanded(!expanded)}
    >
      {/* Profile Photo Circle */}
      <div style={{
        width: '100px',
        height: '100px',
        borderRadius: '50%',
        backgroundColor: 'rgba(255, 255, 255, 0.2)',
        margin: '0 auto 15px auto',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        fontSize: '32px',
        color: 'rgba(255, 255, 255, 0.5)',
        overflow: 'hidden'
      }}>
        <img style={{width: '100px',height: '100px'}} src={member.img}></img>
      </div>

      {/* Name */}
      <h4 style={{
        margin: '0 0 5px 0',
        fontSize: '1.1rem',
        color: 'white'
      }}>
        {member.name}
      </h4>

      {/* Affiliation */}
      <p style={{
        margin: '0 0 10px 0',
        fontSize: '0.85rem',
        color: 'rgba(255, 255, 255, 0.7)',
        fontStyle: 'italic'
      }}>
        {member.affiliation}
      </p>

      {/* Link Icons */}
      <div style={{
        display: 'flex',
        justifyContent: 'center',
        gap: '12px',
        marginBottom: '10px',
      }}>
        {member.googleScholar && (
          <a
            href={member.googleScholar}
            target="_blank"
            rel="noopener noreferrer"
            onClick={(e) => e.stopPropagation()}
            style={{
              color: 'rgba(255, 255, 255, 0.7)',
              transition: 'color 0.2s ease',
            }}
            title="Google Scholar"
          >
            <ScholarIcon />
          </a>
        )}
        {member.website && (
          <a
            href={member.website}
            target="_blank"
            rel="noopener noreferrer"
            onClick={(e) => e.stopPropagation()}
            style={{
              color: 'rgba(255, 255, 255, 0.7)',
              transition: 'color 0.2s ease',
            }}
            title="Personal Website"
          >
            <WebsiteIcon />
          </a>
        )}
      </div>

      {/* Contact */}
      {member.contact && (
        <p style={{
          margin: '0 0 10px 0',
          fontSize: '0.8rem',
          color: 'rgba(255, 255, 255, 0.6)',
        }}>
          {member.contact}
        </p>
      )}

      {/* Expandable Description */}
      <div style={{
        maxHeight: expanded ? '250px' : '0',
        overflow: 'hidden',
        transition: 'max-height 0.3s ease',
      }}>
        <p style={{
          margin: '10px 0 0 20px',
          fontSize: '0.9rem',
          color: 'rgba(255, 255, 255, 0.8)',
          textAlign: 'left',
          lineHeight: '1.5'
        }}>
          {member.description}
        </p>
      </div>

      {/* Expand indicator */}
      <span style={{
        fontSize: '0.75rem',
        color: 'rgba(255, 255, 255, 0.5)',
        marginTop: '8px',
        display: 'block'
      }}>
        {/* {expanded ? 'V' : null} */}
      </span>
    </div>
  );
}

function Timeline() {
  const launchDate = new Date('2026-04-01');
  const deadlineDate = new Date('2026-06-15');
  const [now, setNow] = useState(new Date());

  useEffect(() => {
    const timer = setInterval(() => setNow(new Date()), 60000);
    return () => clearInterval(timer);
  }, []);

  const getDaysUntil = (targetDate) => {
    const diff = targetDate - now;
    return Math.ceil(diff / (1000 * 60 * 60 * 24));
  };

  const daysToLaunch = getDaysUntil(launchDate);
  const daysToDeadline = getDaysUntil(deadlineDate);
  const hasLaunched = daysToLaunch <= 0;
  const hasClosed = daysToDeadline <= 0;

  const getBoxStyle = () => {
    if (hasClosed) {
      return {
        background: 'linear-gradient(135deg, rgba(226, 74, 74, 0.15), rgba(226, 74, 74, 0.05))',
        border: '1px solid rgba(226, 74, 74, 0.3)',
      };
    }
    if (hasLaunched) {
      return {
        background: 'linear-gradient(135deg, rgba(74, 180, 226, 0.15), rgba(74, 180, 226, 0.05))',
        border: '1px solid rgba(74, 180, 226, 0.3)',
      };
    }
    return {
      background: 'linear-gradient(135deg, rgba(74, 226, 144, 0.15), rgba(74, 226, 144, 0.05))',
      border: '1px solid rgba(74, 226, 144, 0.3)',
    };
  };

  const getLineGradient = () => {
    if (hasClosed) {
      return 'linear-gradient(90deg, rgba(74, 226, 144, 0.8), rgba(226, 74, 74, 0.8))';
    }
    return 'linear-gradient(90deg, rgba(74, 226, 144, 0.8), rgba(74, 180, 226, 0.8))';
  };

  return (
    <>
      <h2>Timeline</h2>

      <div style={{
        ...getBoxStyle(),
        borderRadius: '12px',
        padding: '30px 40px',
        marginBottom: '40px',
        maxWidth: '600px'
      }}>
        {/* Timeline row with circles */}
        <div style={{
          display: 'flex',
          alignItems: 'flex-start',
          justifyContent: 'space-between',
          position: 'relative',
        }}>
          {/* Connecting line - absolute positioned behind circles */}
          <div style={{
            position: 'absolute',
            top: '40px',
            left: '40px',
            right: '40px',
            height: '3px',
            background: getLineGradient(),
            zIndex: 0,
          }} />

          {/* Launch Date */}
          <div style={{
            display: 'flex',
            flexDirection: 'column',
            alignItems: 'center',
            zIndex: 1,
          }}>
            <div style={{
              width: '80px',
              height: '80px',
              borderRadius: '50%',
              background: '#1a2332',
              border: '3px solid rgba(74, 226, 144, 0.8)',
              display: 'flex',
              flexDirection: 'column',
              alignItems: 'center',
              justifyContent: 'center',
              boxShadow: hasLaunched ? '0 0 15px rgba(74, 226, 144, 0.4)' : 'none',
            }}>
              <span style={{ fontSize: '1.5rem', fontWeight: 'bold', color: '#4ae290' }}>1</span>
              <span style={{ fontSize: '0.7rem', color: 'rgba(255,255,255,0.7)', textTransform: 'uppercase' }}>Apr</span>
            </div>
            <span style={{
              fontSize: '0.8rem',
              fontWeight: '600',
              color: '#4ae290',
              textTransform: 'uppercase',
              letterSpacing: '1px',
              marginTop: '10px',
            }}>Launch</span>
          </div>

          {/* Close Date */}
          <div style={{
            display: 'flex',
            flexDirection: 'column',
            alignItems: 'center',
            zIndex: 1,
          }}>
            <div style={{
              width: '80px',
              height: '80px',
              borderRadius: '50%',
              background: '#1a2332',
              border: `3px solid ${hasClosed ? 'rgba(226, 74, 74, 0.8)' : 'rgba(74, 180, 226, 0.8)'}`,
              display: 'flex',
              flexDirection: 'column',
              alignItems: 'center',
              justifyContent: 'center',
              boxShadow: hasClosed ? '0 0 15px rgba(226, 74, 74, 0.4)' : (hasLaunched ? '0 0 15px rgba(74, 180, 226, 0.3)' : 'none'),
            }}>
              <span style={{ fontSize: '1.5rem', fontWeight: 'bold', color: hasClosed ? '#e24a4a' : '#4ab4e2' }}>15</span>
              <span style={{ fontSize: '0.7rem', color: 'rgba(255,255,255,0.7)', textTransform: 'uppercase' }}>Jun</span>
            </div>
            <span style={{
              fontSize: '0.8rem',
              fontWeight: '600',
              color: hasClosed ? '#e24a4a' : '#4ab4e2',
              textTransform: 'uppercase',
              letterSpacing: '1px',
              marginTop: '10px',
            }}>Deadline</span>
          </div>
        </div>

        {/* Countdown / Status below */}
        <div style={{
          textAlign: 'center',
          marginTop: '-50px',
          paddingTop: '15px',
          // borderTop: '1px solid rgba(255,255,255,0.1)',
        }}>
          {hasClosed ? (
            <span style={{
              fontSize: '1rem',
              fontWeight: 'bold',
              color: '#e24a4a',
            }}>
              Challenge Closed
            </span>
          ) : (
            <>
              <span style={{ color: 'rgba(255,255,255,0.6)', fontSize: '0.85rem' }}>
                {hasLaunched ? 'Submissions close in ' : 'Challenge launches in '}
              </span>
              <span style={{
                fontSize: '1.1rem',
                fontWeight: 'bold',
                color: hasLaunched ? '#4ab4e2' : '#4ae290',
              }}>
                {hasLaunched ? daysToDeadline : daysToLaunch} days
              </span>
            </>
          )}
        </div>
      </div>
    </>
  );
}

export default function BioDCASE() {
  return (
    <div style={{
      padding: '40px',
      maxWidth: '900px',
      margin: '0 auto',
      color: 'white'
    }}>
        <h2 style={{fontSize: '18px', color: 'grey'}}>BioDCASE 2026</h2>
        <h1>Active Learning for Bioacoustics</h1>

        <tag>Active Learning, Bioacoustics</tag>

        <p>A fundamental challenge across bioacoustics domains (terrestrial and marine) is the annotation of unlabelled data. Passive acoustic monitoring systems generate vast amounts of data, but only a small portion can be feasibly annotated by expert human annotators. Since model performance depends heavily on the quality and quantity of labelled data, this raises the following research question:</p> 
        <p style={{fontSize: '18px', color: "rgb(171 171 171)", padding: '0px 100px 0px 10px'}}>Given vast amounts of raw acoustic data and limited annotation resources, which data should be prioritised for labelling?</p>
        <p>Active learning (AL) is a critical strategy for scaling bioacoustic monitoring. AL is an iterative method of data selection, annotation and model training also often within a human-in-the-loop framework. Fundamentally, AL aims to optimise for a learning objective (e.g. model performance) using less labeled data minimising annotation requirements. Participants will design an active learning strategy (acquisition function) to maximise training efficiency across batches of multi-label data considering informativeness quantification, diversification, long-tail performance and cross-domain generalisation.</p>

        <h3>About BioDCASE</h3>
        <p>BioDCASE (Evaluation & Benchmarking in Automated Bioacoustics) is an initiative focused on advancing research in computational bioacoustics through annual challenges and workshops. This year the <em>Active learning for Bioacoustics</em> (Task 4) challenge will be running.</p>

        <p>Learn more about this and other challenges <a href='https://biodcase.github.io/challenge2026/summary' target=''>here</a>.</p>

        <Timeline />

        <h2>Organising Team</h2>
        <div style={{
          display: 'flex',
          flexWrap: 'wrap',
          gap: '20px',
          justifyContent: 'flex-start',
          marginTop: '20px'
        }}>
          {teamMembers.map((member) => (
            <TeamMemberCard key={member.id} member={member} />
          ))}
        </div>
    </div>
  );
}