#include <bits/stdc++.h>
#define int long long
#define i128 __int128
#define ld long double
#define ull unsigned long long
#define ff(x, y, z) for (int(x) = (y); (x) <= (z); ++(x))
#define ffg(x, y, z) for (int(x) = (y); (x) >= (z); --(x))
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
    int n, m;
    cin >> n >> m;
    vector<int> e[3 * n + 1];
    vector<int> du(3 * n + 1, 0);
    bool flag = true;
    while (m--)
    {
        int x, y, c;
        cin >> x >> y >> c;
        int t1 = (y + 1) / 2;
        int t2 = 2 * n - x + 1;
        int t3 = 2 * n + x - y / 2;
        if (c == t1)
        {
            e[t2].eb(t1);
            e[t3].eb(t1);
            du[t1] += 2;
        }
        else if (c == t2)
        {
            e[t1].eb(t2);
            e[t3].eb(t2);
            du[t2] += 2;
        }
        else if (c == t3)
        {
            e[t1].eb(t3);
            e[t2].eb(t3);
            du[t3] += 2;
        }
        else
            flag = false;
    }
    int cnt = 0;
    if (!flag)
        cout << "No" << endl;
    else
    {
        queue<int> q;
        for (int i = 1; i <= 3 * n; i++)
        {
            if (du[i] == 0)
            {
                q.push(i);
                cnt++;
            }
        }
        while (!q.empty())
        {
            int x = q.front();
            q.pop();
            for (int i : e[x])
            {
                du[i]--;
                if (du[i] == 0)
                {
                    q.push(i);
                    cnt++;
                }
            }
        }
        if (cnt == 3 * n)
            cout << "Yes" << endl;
        else
            cout << "No" << endl;
    }
}
signed main()
{
    ios::sync_with_stdio(false);
    cin.tie(0);
    cout.tie(0);
    int T = 1;
    cin >> T;
    while (T--)
    {
        solve();
    }
    return 0;
}