import $ from 'jquery'
import React from 'react'
import { Redirect, Route, Switch } from 'react-router-dom'

import AppHome from './app_home'
import BfbAd from './bfb_ad'
import BfbPopupAd from './bfb_popup_ad'
import FinalSayAd from './final_say_ad'
import LegalLinks from './legal_links'
import PostcodeAreaInfo from './postcode_area_info'
import PostcodeInfo from './postcode_info'
import Map from './map'
import Nav from './nav'
import PageviewTracker from './pageview_tracker'
import SignsAd from './signs/ad'
import SignInfo from './signs/info'

const SEARCH_PATH = '/search'
const POSTCODE_PATH = '/postcode/:outwardCode/:inwardCode'
const POSTCODE_AREA_PATH = '/area/:postcodeArea'
const SIGN_PATH = '/sign/:signId'

class App extends React.Component {
  constructor(props) {
    super(props)

    this.bottomRef = React.createRef()
    this.state = {
      isClient: false,
      infoOnBottom: false
    }
  }

  componentDidMount() {
    // Trick to avoid rehydration mismatch. The server always renders the home
    // page, but the user might be loading with a path in the anchor; rehydrate
    // with the home page first, then re-render with actual content. This
    // approach is from https://reactjs.org/docs/react-dom.html#hydrate .
    this.setState({ isClient: true })

    // Trick to render the info bar below the map on mobile.
    // The bottom row's visibility is controlled by a media query.
    const handleResize = () => {
      const infoOnBottom = $(this.bottomRef.current).is(':visible')
      if (this.state.infoOnBottom !== infoOnBottom)
        this.setState({ infoOnBottom })
    }
    window.addEventListener('resize', handleResize)
    handleResize()
  }

  render() {
    const { isClient, infoOnBottom } = this.state

    const legacyHashRouterRedirect =
      isClient && this.detectLegacyHashRouterHash()
    if (legacyHashRouterRedirect) return legacyHashRouterRedirect

    const pageviewTracker = isClient && <PageviewTracker />
    const postcodeInfo = <Route path={POSTCODE_PATH} component={PostcodeInfo} />
    const areaInfo = (
      <Route path={POSTCODE_AREA_PATH} component={PostcodeAreaInfo} />
    )
    const signInfo = <Route path={SIGN_PATH} component={SignInfo} />

    return (
      <React.Fragment>
        {pageviewTracker}
        <div className="row no-gutters" id="my-eu-app">
          <div className="col-md-5" id="my-eu-bar">
            <Nav path="/" />
            <div className="container">
              <div className="row">
                <div className="col">
                  <Switch>
                    <Route exact={isClient} path="/" component={AppHome} />
                    <Route path={SEARCH_PATH} component={AppHome} />
                    {!infoOnBottom && postcodeInfo}
                    {!infoOnBottom && areaInfo}
                    {!infoOnBottom && signInfo}
                  </Switch>
                  {!infoOnBottom && <SignsAd />}
                  {!infoOnBottom && <FinalSayAd />}
                  {!infoOnBottom && <BfbAd />}
                  {!infoOnBottom && <LegalLinks />}
                </div>
              </div>
            </div>
          </div>
          <div className="col-md-7">
            <Switch>
              <Route exact path="/" component={Map} />
              <Route path={SEARCH_PATH} component={Map} />
              <Route path={POSTCODE_PATH} component={Map} />
              <Route path={POSTCODE_AREA_PATH} component={Map} />
              <Route path={SIGN_PATH} component={Map} />
            </Switch>
          </div>
        </div>
        <div ref={this.bottomRef} className="row no-gutters d-md-none">
          <div className="col">
            <div className="container">
              <div className="row">
                <div className="col">
                  <Switch>
                    {infoOnBottom && postcodeInfo}
                    {infoOnBottom && areaInfo}
                    {infoOnBottom && signInfo}
                  </Switch>
                  {infoOnBottom && <SignsAd />}
                  {infoOnBottom && <FinalSayAd />}
                  {infoOnBottom && <BfbAd />}
                  {infoOnBottom && <LegalLinks />}
                </div>
              </div>
            </div>
          </div>
        </div>
        <BfbPopupAd />
      </React.Fragment>
    )
  }

  // We used to use HashRouter. If we get a hash router path, redirect to that
  // path without the hash.
  detectLegacyHashRouterHash() {
    if (!window.location.hash.startsWith('#/')) return null
    return <Redirect to={window.location.hash.slice(1)} />
  }
}
export default App
