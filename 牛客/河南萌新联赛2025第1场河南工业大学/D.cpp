#include <bits/stdc++.h>
#define int long long
#define ull unsigned long long
#define i128 __int128
#define ld long double
#define ff(x, y, z) for (int(x) = (y); (x) <= (z); ++(x))
#define ffg(x, y, z) for (int(x) = (y); (x) >= (z); --(x))
#define pb push_back
#define eb emplace_back
#define pii pair<int, int>
#define vc vector
#define vi vector<int>
#define vvi vector<vi>
#define fi first
#define se second
#define all(x) x.begin(), x.end()
#define all1(x) x.begin() + 1, x.end()
#define INF 0x7fffffffffffffff
#define inf 0x7fffffff
// #define endl endl << flush
#define endl '\n'
#define WA AC
#define TLE AC
#define MLE AC
#define RE AC
#define CE AC
using namespace std;
const string AC = "Accepted";
const int MOD = 1e9 + 7;
const int mod = 998244353;
const int N = 1e4 + 6;
vi res(N, 0);
vi need(N, 0);
void Murasame()
{
    int n, q;
    cin >> n >> q;
    int nowtime = 0, now = 1;
    int goal, time;
    int ok = 0;
    vi num;
    vc<pii> ns(n + 1);
    res[0] = 1;
    ff(i, 1, n)
    {
        cin >> ns[i].fi >> ns[i].se;
        need[ns[i].fi] = ns[i].se;
    }
    goal = ns[1].se;
    time = ns[1].fi;
    need[time] = 0;
    while (nowtime <= n)
    {
        if (goal == res[nowtime])
            ok = 0;
        else if (goal < res[nowtime])
            ok = -1;
        else
            ok = 1;
        ff(i, nowtime, time)
        {
            if (i == 0)
                continue;
            res[i] = res[i - 1];
        }
        nowtime = max(time, nowtime);
        for (int i = nowtime + 1;; i++)
        {
            if (need[i] != 0)
            {
                num.pb(need[i]);
                need[i] = 0;
            }
            res[i] = res[i - 1] + 1;
            if (res[i] == goal)
            {
                nowtime = i;
                break;
            }
        }
    he:
        if (!num.empty())
        {
            if (ok == 0)
            {
                goal = num[0];
                num.erase(num.begin());
            }
            else if (ok == 1)
            {
                int it = 0;
                for (int i = 0; i < num.size(); i++)
                {
                    if (goal >= num[i])
                    {
                        goal = num[i];
                        it = i;
                        num.erase(num.begin() + it);
                        break;
                    }
                }
                if (!it)
                {
                    goal = num[0];
                    num.erase(num.begin());
                }
            }
            else
            {
                int it = 0;
                for (int i = 0; i < num.size(); i++)
                {
                    if (goal <= num[i])
                    {
                        goal = num[i];
                        it = i;
                        num.erase(num.begin() + it);
                        break;
                    }
                }
                if (!it)
                {
                    goal = num[0];
                    num.erase(num.begin());
                }
            }
        }
        else
        {
            for (int i = nowtime + 1; i <= n; i++)
            {
                res[i] = res[i - 1];
                time = i;
                if (need[i] != 0)
                {
                    num.pb(need[i]);
                    need[i] = 0;
                    nowtime = time;
                    goto he;
                }
            }
            break;
        }
    }
    while (q--)
    {
        int p;
        cin >> p;
        cout << res[p] << endl;
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
        Murasame();
    }
    return 0;
}