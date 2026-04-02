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
const int N = 1e6 + 6;
void Murasame()
{
    int n;
    cin >> n;
    string s;
    cin >> s;
    int cnt = 0;
    vvi g(n);
    vi du(n, 0);
    ffg(i, n - 1, 0)
    {
        if (s[i] == '0')
        {
            cnt++;
            g[i].pb((i + 1) % n);
            du[(i + 1) % n]++;
        }
        else
        {
            g[(i + 1) % n].pb(i);
            du[i]++;
        }
    }
    if (cnt == n || cnt == 0)
    {
        cout << -1 << endl;
        return;
    }
    queue<int> q;
    ff(i, 0, n - 1)
    {
        if (du[i] == 0)
        {
            q.push(i);
        }
    }
    vi ans(n);
    cnt = -1;
    while (!q.empty())
    {
        int u = q.front();
        q.pop();
        ans[u] = ++cnt;
        for (int v : g[u])
        {
            du[v]--;
            if (du[v] == 0)
            {
                q.push(v);
            }
        }
    }
    ff(i, 0, n - 1)
    {
        cout << ans[i] << " ";
    }
    cout << endl;
}
signed main()
{
    ios::sync_with_stdio(false);
    cin.tie(0);
    cout.tie(0);
    int _T = 1;
    // cin >> _T;
    while (_T--)
    {
        Murasame();
    }
    return 0;
}
// #include <bits/stdc++.h>
// #define int long long
// #define ull unsigned long long
// #define i128 __int128
// #define ld long double
// #define ff(x, y, z) for (int(x) = (y); (x) <= (z); ++(x))
// #define ffg(x, y, z) for (int(x) = (y); (x) >= (z); --(x))
// #define pb push_back
// #define eb emplace_back
// #define pii pair<int, int>
// #define vc vector
// #define vi vector<int>
// #define vvi vector<vi>
// #define fi first
// #define se second
// #define all(x) x.begin(), x.end()
// #define all1(x) x.begin() + 1, x.end()
// #define INF 0x7fffffffffffffff
// #define inf 0x7fffffff
// // #define endl endl << flush
// #define endl '\n'
// #define WA AC
// #define TLE AC
// #define MLE AC
// #define RE AC
// #define CE AC
// using namespace std;
// const string AC = "Accepted";
// const int MOD = 1e9 + 7;
// const int mod = 998244353;
// const int N = 1e6 + 6;
// void Murasame()
// {
//     int n;
//     cin >> n;
//     string s;
//     cin >> s;
//     vi a(n, 0);
//     int cnt = 0;
//     ff(i, 0, n - 1)
//     {
//         if (s[i] == '0')
//             cnt++;
//     }
//     if (cnt == n || cnt == 0)
//     {
//         cout << -1 << endl;
//         return;
//     }
//     ffg(i, n - 1, 0)
//     {
//         if (s[i] == '0')
//         {
//             a[i] = a[(i + 1) % n] - 1;
//         }
//         else
//         {
//             a[i] = a[(i + 1) % n] + 1e9;
//         }
//         // cout<<i<<" "<<(i+1+n)%n<<endl;
//     }
//     vi b = a;
//     sort(all(b));
//     b.erase(unique(all(b)), b.end());
//     vi c(n, 0);
//     ff(i, 0, n - 1)
//     {
//         int j = lower_bound(all(b), a[i]) - b.begin();
//         c[i] = j;
//         // cout << a[i] << " ";
//     }
//     ff(i, 0, n - 1)
//     {
//         cout << c[i] << " ";
//     }
//     cout << endl;
// }
// signed main()
// {
//     ios::sync_with_stdio(false);
//     cin.tie(0);
//     cout.tie(0);
//     int _T = 1;
//     // cin >> _T;
//     while (_T--)
//     {
//         Murasame();
//     }
//     return 0;
// }