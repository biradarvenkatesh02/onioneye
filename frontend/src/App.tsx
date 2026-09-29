import { useEffect, useState } from 'react'
import { LuCamera, LuLayoutDashboard, LuHistory, LuCpu } from 'react-icons/lu'
import Inspect from './pages/Inspect'
import Result from './pages/Result'
import History from './pages/History'
import Dashboard from './pages/Dashboard'
import Models from './pages/Models'

// Tiny hash router:  #/   #/r/<id>   #/history   #/dashboard   #/models
function useRoute() {
  const [hash, setHash] = useState(window.location.hash || '#/')
  useEffect(() => {
    const on = () => { setHash(window.location.hash || '#/'); window.scrollTo(0, 0) }
    window.addEventListener('hashchange', on)
    return () => window.removeEventListener('hashchange', on)
  }, [])
  return hash
}

const NAV = [
  { href: '#/', label: 'Inspect', short: 'Inspect', icon: LuCamera },
  { href: '#/dashboard', label: 'Dashboard', short: 'Dashboard', icon: LuLayoutDashboard },
  { href: '#/history', label: 'Reports', short: 'Reports', icon: LuHistory },
  { href: '#/models', label: 'AI models', short: 'Models', icon: LuCpu },
]

export default function App() {
  const route = useRoute()
  const resultId = route.startsWith('#/r/') ? route.slice(4) : null
  const active = (href: string) => (href === '#/' ? route === '#/' || !!resultId : route.startsWith(href))

  let page = <Inspect />
  if (resultId) page = <Result id={resultId} />
  else if (route.startsWith('#/history')) page = <History />
  else if (route.startsWith('#/dashboard')) page = <Dashboard />
  else if (route.startsWith('#/models')) page = <Models />

  return (
    <div className="shell">
      <aside className="side">
        <a href="#/" className="logo">
          <img className="logo-mark" src="/favicon.svg" alt="" />
          <div><b>OnionEye</b><span>AI quality grading</span></div>
        </a>
        {NAV.map(n => (
          <a key={n.href} href={n.href} className={`navlink ${active(n.href) ? 'active' : ''}`}>
            <n.icon /> {n.label}
          </a>
        ))}
        <div className="side-foot">
          SIH 2026 · PS SIH26031<br />Dept. of Consumer Affairs<br />Team 200_OK
        </div>
      </aside>
      <main className="main">{page}</main>
      <nav className="bottomnav">
        {NAV.map(n => (
          <a key={n.href} href={n.href} className={active(n.href) ? 'active' : ''}><n.icon />{n.short}</a>
        ))}
      </nav>
    </div>
  )
}
