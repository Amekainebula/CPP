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
const int N = 1e5 + 6;
bool isprime[N];
int prime[N];
int cnt = 0;
void euler()
{
    isprime[0] = isprime[1] = 1;
    for (int i = 2; i <= N; i++)
    {
        if (!isprime[i])
            prime[++cnt] = i;
        for (int j = 1; j <= cnt && i * prime[j] <= N; j++)
        {
            isprime[i * prime[j]] = 1;
            if (i % prime[j] == 0)
                break;
        }
    }
}
void Murasame()
{
    euler();
    int n;
    cin >> n;
    ff(i, 1, n)
    {
        cout<<i*prime[500+i]<<" ";
    }
    cout<<endl;
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