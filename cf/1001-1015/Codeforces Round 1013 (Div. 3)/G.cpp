#include <bits/stdc++.h>
#define int long long
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
    int s, k;
    cin >> s >> k;
    if (s % k == 0)
    {
        cout << k << endl;
    }
    else if (s > k * k)
    {
        cout << max(1LL, k - 2) << endl;
    }
    else
    {
        queue<array<int, 3>> q;
        q.push({0, k + 1, -1});
        int now = 0;
        vector<int> vis(s + 1);
        while (!q.empty())
        {
            auto [pos, val, d] = q.front();
            q.pop();
            d = -d;
            if (val != 1)
                val--;
            if (val != now)
            {
                now = val;
                fill(vis.begin(), vis.end(), 0);
            }
            int x = 1;
            while (1)
            {
                int nxt = pos + d * val * x;
                if (nxt < 0 || nxt > s)
                    break;
                if (nxt == s)
                {
                    cout << val << endl;
                    return;
                }
                if (!vis[nxt])
                {
                    q.push({nxt, val, d}), vis[nxt] = 1;
                }
                x++;
            }
        }
    }
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