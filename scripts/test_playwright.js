const {chromium}=require('playwright');
(async()=>{
  let browser;
  try {
    browser = await chromium.launch({args:['--no-sandbox','--disable-setuid-sandbox']});
    const page = await browser.newPage();
    console.log('Playwright: navigating to', process.env.STREAMLIT_APP_URL);
    const response = await page.goto(process.env.STREAMLIT_APP_URL, {waitUntil:'domcontentloaded', timeout:120000});
    const currentURL = response.url();
    if(/share\.streamlit\.io|\/-\/login/.test(currentURL)){
      console.log('Playwright: redirected to login page; nothing further to do.');
    } else {
      await page.waitForSelector('section[data-testid="stApp"]', {timeout:120000});
      await page.waitForTimeout(5000);
      console.log('Playwright: visit complete');
    }
  } catch(e) {
    console.log('Playwright error:', e && e.message ? e.message : e);
  } finally {
    if(browser){
      try{await browser.close();}catch(_){ }
    }
  }
})();
