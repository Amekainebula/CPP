#include <bits/stdc++.h>
// #define int long long
#define ull unsigned long long
#define i128 __int128
#define ld long double
#define ff(x, y, z) for (int(x) = (y); (x) <= (z); ++(x))
#define ffg(x, y, z) for (int(x) = (y); (x) >= (z); --(x))
#define pb push_back
#define eb emplace_back
#define vc vector
#define pii pair<int, int>
#define mpr make_pair
#define fi first
#define se second
#define all(x) x.begin(), x.end()
#define INF 0x7fffffffffffffff
#define inf 0x7fffffff
#define endl '\n'
#define WA AC
#define TLE AC
#define MLE AC
#define RE AC
#define CE AC
using namespace std;
const string AC = "Accepted";

void solve()
{
    int n, u, k, hq;
    cin >> n >> u >> k >> hq;
    vc<int> atk(n + 1);
    vc<array<int, 3>> idhp(n + 1);
    vector<bool> live(n + 1, true);
    priority_queue<pii> pq;
    for (int i = 1; i <= n; i++)
    {
        cin >> atk[i];
        pq.push({atk[i], i});
        int hp;
        cin >> hp;
        idhp[i] = {hp, atk[i], i};
    }
    auto cmp = [](const array<int, 3> &a, const array<int, 3> &b)
    {
        if (a[0] != b[0])
            return a[0] < b[0];
        else if (a[1] != b[1])
            return a[1] < b[1];
        else
            return a[2] < b[2];
    };
    sort(idhp.begin() + 1, idhp.end(), cmp);
    int ans = 0;
    for (int i = 1; i <= n && hq > 0; i++)
    {
        int cnt = 1 + max(0, (int)(idhp[i][0] - u) / (u / 2));
        pii p = pq.top();
        while (!live[p.se])
        {
            pq.pop();
            p = pq.top();
        }
        if (cnt > k)
        {
            hq -= p.fi * k;
        }
        else
        {
            hq -= p.fi * (cnt - 1);
            if (hq <= 0)
                break;
            if (idhp[i][2] == p.se)
            {
                live[idhp[i][2]] = false;
                pq.pop();
                p = pq.top();
                hq -= p.fi;
                ans++;
            }
            else
            {
                hq -= p.fi;
                live[idhp[i][2]] = false; 
                ans++;
            }
        }
    }
    cout << ans << endl;
}
signed main()
{
    ios::sync_with_stdio(false);
    cin.tie(0);
    cout.tie(0);
    int _T = 1;
    //
    cin >> _T;
    while (_T--)
    {
        solve();
    }
    return 0;
}