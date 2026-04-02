#include <bits/stdc++.h>
#define int long long
#define i128 __int128
#define ld long double
#define ull unsigned long long
#define lowbit(x) (x & -x)
#define pb push_back
#define eb emplace_back
#define pii pair<int, int>
#define mpr make_pair
#define fi first
#define se second
#define all(x) x.begin(), x.end()
#define rall(x) x.rbegin(), x.rend()
#define sz(x) (int)(x).size()
#define INF 0x7fffffffffffffff
#define endl '\n'
using namespace std;
vector<int> s[1000006];
void solve()
{
    int n;
    cin >> n;
    set<int> st;
    int maxx = -1;
    for (int i = 1; i <= n; i++)
    {
        int x;
        cin >> x;
        s[x].pb(i);
        maxx = max(maxx, x);
        st.insert(x);
    }
    int ans = INF;
    for (auto i : st)
    {
        //cout<<sz(s[i])<<endl;
        if (sz(s[i]) > 1)
        {
            for (int j = 0; j < sz(s[i]) - 1; j++)
            {
                ans = min(ans, s[i][j + 1] - s[i][j] + 1);
            }
        }
    }
    ans==INF?cout<<-1<<endl:cout<<ans<<endl;
}
signed main()
{
    ios::sync_with_stdio(false);
    cin.tie(0);
    cout.tie(0);
    int T = 1;
    // cin >> T;
    while (T--)
    {
        solve();
    }
    return 0;
}