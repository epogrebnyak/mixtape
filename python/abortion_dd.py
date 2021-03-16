import numpy as np
import pandas as pd
import plotnine as p
import statsmodels.api as sm
import statsmodels.formula.api as smf


def read_data(filename):
    url = "https://raw.github.com/scunning1975/mixtape/master/" + filename
    return pd.read_stata(url)


# import data
abortion = read_data("abortion.dta")
abortion = abortion[~pd.isnull(abortion.lnr)]
abortion_bf15 = abortion[abortion.bf15 == 1]

# fit regression
formula = "lnr ~ C(repeal)*C(year) + C(fip) + acc + ir + pi + alcohol + crack + poverty + income + ur"

reg = (smf.wls(formula, 
               data=abortion_bf15, 
               weights=abortion_bf15.totpop.values)
          .fit(cov_type="cluster", 
               cov_kwds={"groups": abortion_bf15.fip.values}, 
               method="pinv"))

print(reg.summary())

# create dataframe for plotting
abortion_df = pd.DataFrame()
abortion_df["sd"] = reg.bse["C(repeal)[T.1.0]:C(year)[T.1986.0]":"C(repeal)[T.1.0]:C(year)[T.2000.0]"]
abortion_df["mean"] = reg.params["C(repeal)[T.1.0]:C(year)[T.1986.0]":"C(repeal)[T.1.0]:C(year)[T.2000.0]"]
abortion_df["year"] = np.arange(1986, 2001)
abortion_df["lb"] = abortion_df["mean"] - abortion_df["sd"] * 1.96
abortion_df["ub"] = abortion_df["mean"] + abortion_df["sd"] * 1.96

# create plot with plotnine
(
    p.ggplot(abortion_df, p.aes(x="year", y="mean"))
    + p.geom_rect(
        p.aes(xmin=1985, xmax=1992, ymin=-np.inf, ymax=np.inf), fill="cyan", alpha=0.01
    )
    + p.geom_point()
    + p.geom_text(p.aes(label="year"), ha="right")
    + p.geom_hline(yintercept=0)
    + p.geom_errorbar(
        p.aes(ymin="lb", ymax="ub"), width=0.2, position=p.position_dodge(0.05)
    )
    + p.labs(title="Estimated effect of abortion legalization on gonorrhea")
)